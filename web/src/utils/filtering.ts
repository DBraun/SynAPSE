import type { DataPoint, ModalityFilter } from '../types';

type FilterNode =
  | { type: 'term'; value: string }
  | { type: 'and'; children: FilterNode[] }
  | { type: 'or'; children: FilterNode[] }
  | { type: 'not'; child: FilterNode };

/**
 * Tokenize a filter expression
 */
function tokenize(expression: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let inQuotes = false;

  for (const char of expression) {
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && char === ' ') {
      if (current) {
        tokens.push(current);
        current = '';
      }
    } else {
      current += char;
    }
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}

/**
 * Parse a filter expression into an AST
 * Supports: AND, OR, NOT, parentheses, quoted strings
 */
function parseExpression(tokens: string[]): FilterNode | null {
  if (tokens.length === 0) return null;

  // Handle OR (lowest precedence)
  const orIndex = tokens.indexOf('OR');
  if (orIndex !== -1) {
    const left = parseExpression(tokens.slice(0, orIndex));
    const right = parseExpression(tokens.slice(orIndex + 1));
    if (left && right) {
      return { type: 'or', children: [left, right] };
    }
  }

  // Handle AND
  const andIndex = tokens.indexOf('AND');
  if (andIndex !== -1) {
    const left = parseExpression(tokens.slice(0, andIndex));
    const right = parseExpression(tokens.slice(andIndex + 1));
    if (left && right) {
      return { type: 'and', children: [left, right] };
    }
  }

  // Handle NOT
  if (tokens[0] === 'NOT' && tokens.length > 1) {
    const child = parseExpression(tokens.slice(1));
    if (child) {
      return { type: 'not', child };
    }
  }

  // Single term
  if (tokens.length === 1) {
    return { type: 'term', value: tokens[0].toLowerCase() };
  }

  // Implicit AND for multiple terms without operators
  const children: FilterNode[] = [];
  for (const token of tokens) {
    if (token !== 'AND' && token !== 'OR' && token !== 'NOT') {
      children.push({ type: 'term', value: token.toLowerCase() });
    }
  }
  if (children.length > 1) {
    return { type: 'and', children };
  }
  if (children.length === 1) {
    return children[0];
  }

  return null;
}

/**
 * Evaluate a filter AST against a code string
 */
function evaluateFilter(node: FilterNode, code: string): boolean {
  const lowerCode = code.toLowerCase();

  switch (node.type) {
    case 'term':
      return lowerCode.includes(node.value);
    case 'and':
      return node.children.every(child => evaluateFilter(child, code));
    case 'or':
      return node.children.some(child => evaluateFilter(child, code));
    case 'not':
      return !evaluateFilter(node.child, code);
    default:
      return true;
  }
}

/**
 * Check if a point matches a text filter expression
 */
export function matchesTextFilter(point: DataPoint, filterExpression: string): boolean {
  if (!filterExpression.trim()) return true;

  const tokens = tokenize(filterExpression);
  const ast = parseExpression(tokens);
  if (!ast) return true;

  return evaluateFilter(ast, point.preset_name);
}

/**
 * Check if a point has all the required tags
 */
export function matchesTags(point: DataPoint, requiredTags: string[]): boolean {
  if (requiredTags.length === 0) return true;
  return requiredTags.every(tag => point.tags.includes(tag));
}

/**
 * Check if a point matches the modality filter
 */
export function matchesModalityFilter(point: DataPoint, modalityFilter: ModalityFilter): boolean {
  if (modalityFilter === 'all') return true;
  return point.modality === modalityFilter;
}

/**
 * Apply text, tag, and modality filters to a dataset
 */
export function applyFilters(
  data: DataPoint[],
  textFilter: string,
  activeTags: string[],
  modalityFilter: ModalityFilter
): DataPoint[] {
  return data.filter(
    point =>
      matchesTextFilter(point, textFilter) &&
      matchesTags(point, activeTags) &&
      matchesModalityFilter(point, modalityFilter)
  );
}
