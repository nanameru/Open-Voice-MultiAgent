import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ChatInputProps extends React.HTMLAttributes<HTMLFormElement> {
  onSend?: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, className, disabled, ...props }: ChatInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState<string>('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    props.onSubmit?.(e);
    onSend?.(message);
    setMessage('');
  };

  const isDisabled = disabled || message.trim().length === 0;

  useEffect(() => {
    if (disabled) return;
    // when not disabled refocus on input
    inputRef.current?.focus();
  }, [disabled]);

  return (
    <form
      {...props}
      onSubmit={handleSubmit}
      className={cn('flex items-center gap-3 text-sm', className)}
    >
      <div className="flex-1 flex items-center gap-2 rounded-full bg-transparent border border-white/20 px-5 py-2.5">
        <input
          autoFocus
          ref={inputRef}
          type="text"
          value={message}
          disabled={disabled}
          placeholder="なんでも聞いてみて。"
          onChange={(e) => setMessage(e.target.value)}
          className="flex-1 bg-transparent text-white placeholder:text-white/40 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
        />
      </div>
      <Button
        size="sm"
        type="submit"
        variant={isDisabled ? 'secondary' : 'primary'}
        disabled={isDisabled}
        className="font-mono flex-shrink-0"
      >
        SEND
      </Button>
    </form>
  );
}
