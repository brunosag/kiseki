// The little Markdown that report sections are written in: paragraphs, pipe tables, and
// **bold** and `code` spans. Parsed to data and rendered as React elements, never as
// HTML, so a report's text cannot inject markup.

export type Block =
  | { kind: "paragraph"; text: string }
  | { kind: "table"; head: string[]; rows: string[][] };

export type Span = string | { bold: string } | { code: string };

const cells = (line: string) =>
  line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());

const isRule = (line: string) => /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$/.test(line.trim());

export function blocks(text: string): Block[] {
  const out: Block[] = [];
  const lines = text.split("\n");
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) out.push({ kind: "paragraph", text: paragraph.join(" ") });
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().startsWith("|") && isRule(lines[i + 1] ?? "")) {
      flush();
      const head = cells(line);
      const rows: string[][] = [];
      for (i += 2; i < lines.length && lines[i].trim().startsWith("|"); i++) rows.push(cells(lines[i]));
      i--;
      out.push({ kind: "table", head, rows });
    } else if (!line.trim()) flush();
    else paragraph.push(line.trim());
  }
  flush();
  return out;
}

export function inline(text: string): Span[] {
  const out: Span[] = [];
  const pattern = /\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) out.push(text.slice(last, match.index));
    out.push(match[1] != null ? { bold: match[1] } : { code: match[2] });
    last = match.index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
