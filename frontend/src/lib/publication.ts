export function publicationLabel(status?: string): string {
  return (
    (
      {
        retracted: "Retraction notice reported",
        withdrawn: "Withdrawal notice reported",
        expression_of_concern: "Expression of concern reported",
      } as Record<string, string>
    )[status || ""] || "Retraction status unknown"
  );
}
