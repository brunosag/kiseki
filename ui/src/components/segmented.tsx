// An exclusive choice between views (Accuracy | Loss, Step | Time): coss's segmented
// control on Base UI's toggle group, which never lets the choice go empty.
import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { segmentedControlItemVariants, segmentedControlRootClassName } from "@/lib/segmented-control";
import { cn } from "@/lib/utils";

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: React.ReactNode }[];
  label: string;
  className?: string;
}) {
  return (
    <ToggleGroup
      aria-label={label}
      value={[value]}
      onValueChange={(next) => next[0] && onChange(next[0] as T)}
      className={cn(segmentedControlRootClassName, className)}
    >
      {options.map((option) => (
        <Toggle
          key={option.value}
          value={option.value}
          className={cn(segmentedControlItemVariants({ size: "sm", state: "pressed" }), "text-xs sm:text-xs")}
        >
          {option.label}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}
