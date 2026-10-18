// A compact select that reads as a property chip (Linear's "Status", "Assignee" chips in
// its composer): the current choice, and a menu of the others with optional notes.
import { Select, SelectItem, SelectPopup, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface Choice {
  value: string;
  label: string;
  note?: string;
}

export function Pick({
  label,
  value,
  choices,
  onChange,
  icon,
  className,
}: {
  label: string;
  value: string;
  choices: Choice[];
  onChange: (value: string) => void;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <Select items={choices} value={value} onValueChange={(next) => next != null && onChange(next as string)}>
      <SelectTrigger
        size="sm"
        aria-label={label}
        className={cn("w-auto min-w-0 gap-1.5 text-[13px] sm:text-[13px] [&_[data-slot=select-icon]]:hidden", className)}
      >
        {icon && <span className="text-muted-foreground [&_svg]:size-3.5">{icon}</span>}
        <SelectValue />
      </SelectTrigger>
      <SelectPopup alignItemWithTrigger={false}>
        {choices.map((choice) => (
          <SelectItem key={choice.value} value={choice.value} className="text-[13px] sm:text-[13px]">
            <span className="flex items-baseline gap-2">
              {choice.label}
              {choice.note && <span className="text-muted-foreground text-xs">{choice.note}</span>}
            </span>
          </SelectItem>
        ))}
      </SelectPopup>
    </Select>
  );
}
