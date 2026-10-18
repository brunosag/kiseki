// The frame every view shares: a slim header row (title or breadcrumb, then actions) over
// a scrolling body.
import { ChevronRightIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function PageHeader({ children, actions }: { children: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b px-4">
      <div className="flex min-w-0 flex-1 items-center gap-1.5 text-[13px]">{children}</div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </header>
  );
}

export function Crumbs({ items }: { items: { label: React.ReactNode; href?: string }[] }) {
  return (
    <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5">
      {items.map((item, index) => (
        <span key={index} className="flex min-w-0 items-center gap-1.5">
          {index > 0 && <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />}
          {item.href ? (
            <a href={item.href} className="truncate text-muted-foreground transition-colors hover:text-foreground">
              {item.label}
            </a>
          ) : (
            <span className="truncate font-medium">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}

export function PageBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("min-h-0 flex-1 overflow-y-auto", className)}>{children}</div>;
}

/** A label over a value, for the properties panels. */
export function Property({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] items-baseline gap-2 py-1 text-[13px]">
      <dt className="truncate text-muted-foreground">{label}</dt>
      <dd className="min-w-0 truncate">{children}</dd>
    </div>
  );
}

export function PanelSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b px-5 py-4 last:border-b-0">
      <h3 className="mb-2 font-medium text-muted-foreground text-xs">{title}</h3>
      <dl>{children}</dl>
    </section>
  );
}
