"""Single native parser process slot; terminate on deadline or cancellation."""

import asyncio
import multiprocessing
import threading
import time

from app.config import settings

_SLOT = threading.BoundedSemaphore(1)


def _worker(connection, payload, max_pages):
    try:
        from app.services.pdf_parser import PDFParserService

        parsed = PDFParserService(max_pages=max_pages).parse(
            payload, source_origin="web"
        )
        if sum(len(p.text) for p in parsed.pages) > 1_000_000:
            raise ValueError("Extracted text too large")
        connection.send((True, parsed))
    except Exception:
        connection.send((False, None))
    finally:
        connection.close()


async def parse_temporary_pdf(payload):
    if len(payload) > settings.web_pdf_bytes or not payload.startswith(b"%PDF-"):
        raise ValueError("Invalid temporary PDF")
    if not _SLOT.acquire(blocking=False):
        raise ValueError("Temporary parser busy")
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(writer, payload, settings.web_pdf_pages), daemon=True
    )
    try:
        process.start()
        writer.close()
        deadline = time.monotonic() + settings.web_parser_timeout_seconds
        while time.monotonic() < deadline:
            if reader.poll():
                # Our trusted child serializes bounded parser output, never PDF objects.
                ok, result = await asyncio.wait_for(
                    asyncio.to_thread(reader.recv),
                    max(0.01, deadline - time.monotonic()),
                )
                if not ok:
                    raise ValueError("Temporary PDF parsing failed")
                return result
            if not process.is_alive():
                raise ValueError("Temporary parser exited")
            await asyncio.sleep(0.02)
        raise TimeoutError("Temporary PDF parsing deadline")
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()
        reader.close()
        writer.close()
        _SLOT.release()
