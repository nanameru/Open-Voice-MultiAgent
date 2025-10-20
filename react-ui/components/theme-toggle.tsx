'use client';

import { useEffect, useState } from 'react';
import { MonitorIcon, MoonIcon, SunIcon } from '@phosphor-icons/react';
import type { ThemeMode } from '@/lib/types';
import { THEME_MEDIA_QUERY, THEME_STORAGE_KEY, cn } from '@/lib/utils';

const THEME_SCRIPT = `
  const doc = document.documentElement;
  const theme = localStorage.getItem("${THEME_STORAGE_KEY}") ?? "system";

  // Reset and compute effective theme
  doc.classList.remove("dark", "light");
  const prefersDark = window.matchMedia("${THEME_MEDIA_QUERY}").matches;
  const effectiveTheme = theme === "system" ? (prefersDark ? "dark" : "light") : theme;

  // Apply class and dataset for consumers
  doc.classList.add(effectiveTheme);
  doc.dataset.theme = effectiveTheme;

  // Notify listeners (e.g., Live2D background) early in head
  try { window.dispatchEvent(new CustomEvent("themechange", { detail: effectiveTheme })); } catch {}
`
  .trim()
  .replace(/\n/g, '')
  .replace(/\s+/g, ' ');

function applyTheme(theme: ThemeMode) {
  const doc = document.documentElement;

  doc.classList.remove('dark', 'light');
  localStorage.setItem(THEME_STORAGE_KEY, theme);

  const prefersDark = window.matchMedia(THEME_MEDIA_QUERY).matches;
  const effectiveTheme: ThemeMode = (theme === 'system' ? (prefersDark ? 'dark' : 'light') : theme) as ThemeMode;

  doc.classList.add(effectiveTheme);
  (doc as HTMLElement & { dataset: { theme?: string } }).dataset.theme = effectiveTheme;

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('themechange', { detail: effectiveTheme }));
  }
}

interface ThemeToggleProps {
  className?: string;
}

export function ApplyThemeScript() {
  return <script id="theme-script">{THEME_SCRIPT}</script>;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const [theme, setTheme] = useState<ThemeMode | undefined>(undefined);

  useEffect(() => {
    const storedTheme = (localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode) ?? 'system';

    setTheme(storedTheme);
  }, []);

  function handleThemeChange(theme: ThemeMode) {
    applyTheme(theme);
    setTheme(theme);
  }

  return (
    <div
      className={cn(
        'text-foreground bg-background flex w-full flex-row justify-end divide-x overflow-hidden rounded-full border',
        className
      )}
    >
      <span className="sr-only">Color scheme toggle</span>
      <button
        type="button"
        onClick={() => handleThemeChange('dark')}
        className="cursor-pointer p-1 pl-1.5"
      >
        <span className="sr-only">Enable dark color scheme</span>
        <MoonIcon size={16} weight="bold" className={cn(theme !== 'dark' && 'opacity-25')} />
      </button>
      <button
        type="button"
        onClick={() => handleThemeChange('light')}
        className="cursor-pointer px-1.5 py-1"
      >
        <span className="sr-only">Enable light color scheme</span>
        <SunIcon size={16} weight="bold" className={cn(theme !== 'light' && 'opacity-25')} />
      </button>
      <button
        type="button"
        onClick={() => handleThemeChange('system')}
        className="cursor-pointer p-1 pr-1.5"
      >
        <span className="sr-only">Enable system color scheme</span>
        <MonitorIcon size={16} weight="bold" className={cn(theme !== 'system' && 'opacity-25')} />
      </button>
    </div>
  );
}
