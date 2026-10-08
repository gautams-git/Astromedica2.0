import React, { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';

/**
 * ThemeToggle Component
 * Handles switching between Light and Dark modes.
 * Persists user choice in localStorage.
 * Updates the 'data-theme' attribute on the root element.
 */
const ThemeToggle = () => {
  // 1. State Management
  const [theme, setTheme] = useState('dark');

  // 2. Initial Load: Check localStorage or default to 'dark'
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
  }, []);

  // 3. Persistence & Side Effects: Update root element and localStorage
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'dark' ? 'light' : 'dark'));
  };

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle Theme"
      style={{
        background: 'transparent',
        border: '1px solid var(--border)',
        color: 'var(--gold)',
        width: '40px',
        height: '40px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        padding: '0',
        outline: 'none',
      }}
      className="theme-toggle-btn"
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--gold-dim)';
        e.currentTarget.style.transform = 'scale(1.05)';
        e.currentTarget.style.borderColor = 'var(--gold)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
        e.currentTarget.style.transform = 'scale(1)';
        e.currentTarget.style.borderColor = 'var(--border)';
      }}
    >
      {theme === 'dark' ? (
        <Sun size={20} strokeWidth={2} />
      ) : (
        <Moon size={20} strokeWidth={2} />
      )}
    </button>
  );
};

export default ThemeToggle;
