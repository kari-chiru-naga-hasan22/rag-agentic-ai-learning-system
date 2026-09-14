import React, { useMemo } from 'react';
import katex from 'katex';

interface MathViewProps {
  math: string;
  block?: boolean;
  className?: string;
}

export const MathView: React.FC<MathViewProps> = ({ math, block = false, className = '' }) => {
  const html = useMemo(() => {
    try {
      const trimmed = math.trim();
      return katex.renderToString(trimmed, {
        displayMode: block,
        throwOnError: false,
        strict: false,
      });
    } catch (e) {
      return `<span class="font-mono text-xs text-amber-500">${math}</span>`;
    }
  }, [math, block]);

  return (
    <span
      className={`inline-math ${block ? 'block overflow-x-auto py-2 my-1' : 'inline-block align-middle mx-0.5'} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};
