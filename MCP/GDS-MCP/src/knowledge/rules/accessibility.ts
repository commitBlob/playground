import { ComponentRule, RuleResult } from "../types.js";

export const accessibilityRules: ComponentRule[] = [
  {
    id: "no-unsafe-url-protocols",
    severity: "error",
    description:
      "Links and resource attributes must not use unsafe URL protocols (javascript:, data:, vbscript:)",
    check: (html: string): RuleResult => {
      const attrRegex = /(href|src|action)\s*=\s*(["'])(.*?)\2/gi;
      for (const match of html.matchAll(attrRegex)) {
        const [, attrName, , attrValue] = match;
        if (/^(?:\s*|[^\S\r\n]*)?(?:javascript|data|vbscript)\s*:/i.test(attrValue)) {
          return {
            passed: false,
            message: `Found unsafe ${attrName} value using blocked protocol: "${attrValue}".`,
            suggestion:
              'Use safe URLs only (for example: "/path", "https://...", "mailto:...", "tel:...").',
          };
        }
      }

      return { passed: true, message: "No unsafe URL protocols found." };
    },
  },
  {
    id: "input-must-have-label",
    severity: "error",
    description: "All form inputs must have an associated visible label",
    check: (html: string): RuleResult => {
      const inputs = html.match(/<input[^>]*>/g) || [];
      for (const input of inputs) {
        if (/type=["']hidden["']/.test(input)) continue;
        const idMatch = input.match(/id=["']([^"']+)["']/);
        if (!idMatch) {
          return {
            passed: false,
            message: "Input element is missing an 'id' attribute needed for label association.",
            suggestion: "Add an id to the input and a <label for=\"[id]\"> element.",
          };
        }
        const inputId = idMatch[1];
        const labelPattern = new RegExp(`for=["']${inputId}["']`);
        if (!labelPattern.test(html)) {
          return {
            passed: false,
            message: `Input with id="${inputId}" has no associated label (no label[for="${inputId}"] found).`,
            suggestion: `Add <label class="govuk-label" for="${inputId}"> before the input.`,
          };
        }
      }
      return { passed: true, message: "All inputs have associated labels." };
    },
  },
  {
    id: "error-message-format",
    severity: "error",
    description: "Error messages must include a visually hidden 'Error:' prefix",
    check: (html: string): RuleResult => {
      const hasErrorMessage = /govuk-error-message/.test(html);
      if (!hasErrorMessage) {
        return { passed: true, message: "No error messages present." };
      }
      const hasHiddenPrefix = /govuk-visually-hidden[^>]*>\s*Error:/.test(html);
      if (!hasHiddenPrefix) {
        return {
          passed: false,
          message: "Error message missing visually hidden 'Error:' prefix for screen readers.",
          suggestion:
            'Add <span class="govuk-visually-hidden">Error:</span> at the start of error message content.',
        };
      }
      return { passed: true, message: "Error messages have correct visually hidden prefix." };
    },
  },
  {
    id: "no-placeholder-only-labels",
    severity: "error",
    description: "Inputs must not use placeholder as the sole label",
    check: (html: string): RuleResult => {
      const inputs = html.match(/<input[^>]*placeholder=["'][^"']+["'][^>]*>/g) || [];
      for (const input of inputs) {
        const idMatch = input.match(/id=["']([^"']+)["']/);
        if (!idMatch) {
          return {
            passed: false,
            message: "Input with placeholder has no id — cannot verify label association.",
            suggestion: "Add an id and a visible <label> element.",
          };
        }
        const labelPattern = new RegExp(`for=["']${idMatch[1]}["']`);
        if (!labelPattern.test(html)) {
          return {
            passed: false,
            message: "Input relies on placeholder text without a visible label.",
            suggestion:
              "Placeholder text disappears on input and is not reliably announced. Add a visible <label>.",
          };
        }
      }
      return { passed: true, message: "No placeholder-only inputs detected." };
    },
  },
  {
    id: "error-summary-role",
    severity: "error",
    description: "Error summary must have role=\"alert\" for screen reader announcement",
    check: (html: string): RuleResult => {
      const hasErrorSummary = /govuk-error-summary/.test(html);
      if (!hasErrorSummary) {
        return { passed: true, message: "No error summary present." };
      }
      const hasRoleAlert = /role=["']alert["']/.test(html);
      if (!hasRoleAlert) {
        return {
          passed: false,
          message: "Error summary is missing role=\"alert\".",
          suggestion:
            "Add role=\"alert\" to the error summary container so screen readers announce it immediately.",
        };
      }
      return { passed: true, message: "Error summary has role=\"alert\"." };
    },
  },
  {
    id: "table-header-scope",
    severity: "warning",
    description: "Table header cells should have a scope attribute",
    check: (html: string): RuleResult => {
      const hasTh = /<th[^>]*>/.test(html);
      if (!hasTh) {
        return { passed: true, message: "No table headers present." };
      }
      const thTags = html.match(/<th[^>]*>/g) || [];
      for (const th of thTags) {
        if (!/scope=["']/.test(th)) {
          return {
            passed: false,
            message: "Table header <th> is missing a scope attribute.",
            suggestion:
              'Add scope="col" for column headers or scope="row" for row headers.',
          };
        }
      }
      return { passed: true, message: "All table headers have scope attributes." };
    },
  },
  {
    id: "action-link-context",
    severity: "warning",
    description: "Action links (Change, Remove, Delete) need visually hidden context for screen readers",
    check: (html: string): RuleResult => {
      const actionLinks = html.match(/<a[^>]*>[^<]*<\/a>/g) || [];
      for (const link of actionLinks) {
        const text = link.replace(/<[^>]*>/g, "").trim();
        if (/^(Change|Remove|Delete|Edit)$/i.test(text)) {
          return {
            passed: false,
            message: `Action link text "${text}" lacks context — screen readers will announce just "${text}" without knowing what it relates to.`,
            suggestion: `Add visually hidden context: ${text}<span class="govuk-visually-hidden"> [item name]</span>`,
          };
        }
      }
      return { passed: true, message: "Action links have sufficient context." };
    },
  },
  {
    id: "aria-describedby-on-error",
    severity: "warning",
    description: "Inputs with error messages should reference them via aria-describedby",
    check: (html: string): RuleResult => {
      const hasErrorMessage = /govuk-error-message/.test(html);
      if (!hasErrorMessage) return { passed: true, message: "No error messages to check." };
      const hasInput = /<(input|textarea|select)[^>]*>/.test(html);
      if (!hasInput) return { passed: true, message: "No form inputs present." };
      if (!/aria-describedby/.test(html)) {
        return {
          passed: false,
          message: "Error message present but input lacks aria-describedby referencing it.",
          suggestion: "Add aria-describedby=\"[id]-error\" to the input to link it to the error message.",
        };
      }
      return { passed: true, message: "Input references error via aria-describedby." };
    },
  },
  {
    id: "skip-link-present",
    severity: "warning",
    description: "Pages should include a skip link for keyboard navigation",
    check: (html: string): RuleResult => {
      const hasMain = /<main/.test(html) || /id=["']main-content["']/.test(html);
      if (!hasMain) return { passed: true, message: "No main content area detected (skip link check not applicable)." };
      if (!/govuk-skip-link/.test(html) && !/<a[^>]*skip[^>]*>/.test(html)) {
        return {
          passed: false,
          message: "Page has main content but no skip link detected.",
          suggestion: "Add <a href=\"#main-content\" class=\"govuk-skip-link\">Skip to main content</a> before the header.",
        };
      }
      return { passed: true, message: "Skip link present." };
    },
  },
  {
    id: "textarea-must-have-label",
    severity: "error",
    description: "Textareas must have an associated visible label",
    check: (html: string): RuleResult => {
      const textareas = html.match(/<textarea[^>]*>/g) || [];
      for (const ta of textareas) {
        const idMatch = ta.match(/id=["']([^"']+)["']/);
        if (!idMatch) {
          return { passed: false, message: "Textarea missing id for label association.", suggestion: "Add an id and a matching <label for=\"[id]\">." };
        }
        const labelPattern = new RegExp(`for=["']${idMatch[1]}["']`);
        if (!labelPattern.test(html)) {
          return { passed: false, message: `Textarea id="${idMatch[1]}" has no associated label.`, suggestion: `Add <label class="govuk-label" for="${idMatch[1]}"> before the textarea.` };
        }
      }
      return { passed: true, message: "All textareas have associated labels." };
    },
  },
  {
    id: "select-must-have-label",
    severity: "error",
    description: "Select elements must have an associated visible label",
    check: (html: string): RuleResult => {
      const selects = html.match(/<select[^>]*>/g) || [];
      for (const sel of selects) {
        const idMatch = sel.match(/id=["']([^"']+)["']/);
        if (!idMatch) {
          return { passed: false, message: "Select element missing id for label association.", suggestion: "Add an id and a matching <label for=\"[id]\">." };
        }
        const labelPattern = new RegExp(`for=["']${idMatch[1]}["']`);
        if (!labelPattern.test(html)) {
          return { passed: false, message: `Select id="${idMatch[1]}" has no associated label.`, suggestion: `Add <label class="govuk-label" for="${idMatch[1]}"> before the select.` };
        }
      }
      return { passed: true, message: "All selects have associated labels." };
    },
  },
];
