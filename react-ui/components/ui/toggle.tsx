'use client';

import * as React from 'react';
import { type VariantProps, cva } from 'class-variance-authority';
import * as TogglePrimitive from '@radix-ui/react-toggle';
import { cn } from '@/lib/utils';

const toggleVariants = cva(
  [
    'inline-flex items-center justify-center gap-2 rounded-full',
    'text-sm font-medium whitespace-nowrap',
    'cursor-pointer outline-none transition-[color,border,background-color]',
    'focus-visible:ring-ring/50 focus-visible:ring-[3px] focus-visible:border-ring',
    'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
    'disabled:pointer-events-none disabled:opacity-50 disabled:not-allowed',
    'data-[state=on]:bg-button-selected data-[state=on]:border-button-border-selected',
    "[&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0",
  ],
  {
    variants: {
      variant: {
        default:
          'bg-white/10 backdrop-blur-xl border border-white/20 text-white hover:bg-white/15 focus:bg-white/15 shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
        primary:
          'text-white bg-white/10 backdrop-blur-xl border border-white/20 hover:bg-white/15 focus:bg-white/15 shadow-[0_8px_32px_rgba(0,0,0,0.2)] data-[state=off]:bg-rose-500/25 hover:data-[state=off]:bg-rose-500/35 data-[state=off]:text-rose-100 data-[state=off]:border-rose-400/30 data-[state=off]:shadow-[0_8px_32px_rgba(251,113,133,0.25)]',
        secondary:
          'text-white bg-white/10 backdrop-blur-xl border border-white/20 hover:bg-white/15 focus:bg-white/15 shadow-[0_8px_32px_rgba(0,0,0,0.2)] data-[state=on]:bg-cyan-500/25 hover:data-[state=on]:bg-cyan-500/35 data-[state=on]:text-cyan-50 data-[state=on]:border-cyan-400/30 data-[state=on]:shadow-[0_8px_32px_rgba(34,211,238,0.25)]',
        outline: [
          'border border-white/20 bg-transparent backdrop-blur-xl text-white shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
          'hover:bg-white/10 focus:bg-white/10',
        ],
      },
      size: {
        default: 'h-9 px-2 min-w-9',
        sm: 'h-8 px-1.5 min-w-8',
        lg: 'h-10 px-2.5 min-w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

function Toggle({
  className,
  variant,
  size,
  ...props
}: React.ComponentProps<typeof TogglePrimitive.Root> & VariantProps<typeof toggleVariants>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(toggleVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Toggle, toggleVariants };
