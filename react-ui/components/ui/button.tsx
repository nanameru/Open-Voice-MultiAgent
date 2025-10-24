import * as React from 'react';
import { type VariantProps, cva } from 'class-variance-authority';
import { Slot } from '@radix-ui/react-slot';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  [
    'text-xs font-bold tracking-wider uppercase whitespace-nowrap',
    'inline-flex items-center justify-center gap-2 shrink-0 rounded-full cursor-pointer outline-none transition-colors duration-300',
    'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
    'disabled:pointer-events-none disabled:opacity-50',
    'aria-invalid:ring-destructive/20 aria-invalid:border-destructive dark:aria-invalid:ring-destructive/40 ',
    "[&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0",
  ],
  {
    variants: {
      variant: {
        default: 'bg-white/10 backdrop-blur-xl border border-white/20 text-white hover:bg-white/15 focus:bg-white/15 shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
        destructive: [
          'bg-rose-500/25 backdrop-blur-xl border border-rose-400/30 text-rose-100 shadow-[0_8px_32px_rgba(251,113,133,0.25)]',
          'hover:bg-rose-500/35 focus:bg-rose-500/35 focus-visible:ring-rose-400/20',
          'dark:focus-visible:ring-rose-400/30',
        ],
        outline: [
          'border border-white/20 bg-transparent backdrop-blur-xl text-white shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
          'hover:bg-white/10 hover:text-white',
          'dark:bg-white/5 dark:border-white/20 dark:hover:bg-white/10',
        ],
        primary: 'bg-cyan-500/25 backdrop-blur-xl border border-cyan-400/30 text-cyan-50 hover:bg-cyan-500/35 focus:bg-cyan-500/35 shadow-[0_8px_32px_rgba(34,211,238,0.25)]',
        secondary: 'bg-slate-500/15 backdrop-blur-xl border border-slate-400/25 text-slate-100 hover:bg-slate-500/25 shadow-[0_8px_32px_rgba(100,116,139,0.2)]',
        ghost: 'hover:bg-white/10 hover:text-white dark:hover:bg-white/10',
        link: 'text-blue-300 underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2 has-[>svg]:px-3',
        sm: 'h-8 gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-10 px-6 has-[>svg]:px-4',
        icon: 'size-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<'button'> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot : 'button';

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
