"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getHealth } from "@/lib/api";
import { BookOpen, Columns3, Compass, Library, Moon, Search, Sun, Upload, Users } from "lucide-react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useSyncExternalStore, useState } from "react";

const NAV_ITEMS = [
  { href: "/discover", label: "Discover", icon: Compass },
  { href: "/researchers", label: "Researchers", icon: Users },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/library", label: "Library", icon: Library },
  { href: "/query", label: "Query", icon: Search },
  { href: "/compare", label: "Compare", icon: Columns3 },
];

// Use useSyncExternalStore to avoid hydration mismatch
function useIsMounted() {
  return useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );
}

export function MainNav() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const mounted = useIsMounted();
  const [healthStatus, setHealthStatus] = useState<"ok" | "error" | "loading">("loading");

  // Check health status
  useEffect(() => {
    async function checkHealth() {
      try {
        const health = await getHealth();
        setHealthStatus(health.status === "ok" ? "ok" : "error");
      } catch {
        setHealthStatus("error");
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-2 px-4">
        {/* Logo */}
        <Link href="/discover" className="flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-primary" />
          <span className="font-semibold">FolioTrace</span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link key={href} href={href}>
                <Button
                  variant={isActive ? "secondary" : "ghost"}
                  size="sm"
                  className={cn(
                    "gap-2",
                    isActive && "bg-secondary"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </Link>
            );
          })}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-2">
          {/* Health indicator */}
          <div
            className={cn(
              "h-2 w-2 rounded-full",
              healthStatus === "ok" && "bg-green-500",
              healthStatus === "error" && "bg-red-500",
              healthStatus === "loading" && "bg-yellow-500 animate-pulse"
            )}
            title={
              healthStatus === "ok"
                ? "Backend connected"
                : healthStatus === "error"
                ? "Backend disconnected"
                : "Checking connection..."
            }
          />

          {/* Theme toggle */}
          {mounted && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="h-9 w-9 p-0"
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
              <span className="sr-only">Toggle theme</span>
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
