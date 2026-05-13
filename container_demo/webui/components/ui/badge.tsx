import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center gap-1 rounded-lg px-3 py-1 text-xs font-semibold", {
  variants: {
    variant: {
      default: "bg-slate-950 text-white",
      warm: "bg-orange-100 text-orange-800",
      cool: "bg-sky-100 text-sky-800",
      mint: "bg-emerald-100 text-emerald-800",
      rose: "bg-rose-100 text-rose-800",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
