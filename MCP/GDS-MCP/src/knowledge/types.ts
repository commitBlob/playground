export type ComponentCategory =
  | "form"
  | "navigation"
  | "messaging"
  | "content"
  | "layout"
  | "action";

export interface TemplateSlot {
  name: string;
  required: boolean;
  type: "text" | "html" | "array" | "boolean";
  description: string;
  default?: string;
}

export interface TemplateVariant {
  name: string;
  description: string;
  additionalSlots?: TemplateSlot[];
}

export interface ComponentTemplate {
  baseMarkup: string;
  slots: TemplateSlot[];
  variants: TemplateVariant[];
}

export interface RuleResult {
  passed: boolean;
  message: string;
  suggestion?: string;
}

export interface ComponentRule {
  id: string;
  severity: "error" | "warning";
  description: string;
  check: (html: string) => RuleResult;
}

export interface ComponentDefinition {
  slug: string;
  name: string;
  category: ComponentCategory;
  description: string;
  useCases: string[];
  antiUseCases: string[];
  relatedComponents: string[];
  template: ComponentTemplate;
  accessibilityRules: ComponentRule[];
}

export interface ScoredMatch {
  component: ComponentDefinition;
  score: number;
  reason: string;
}

export interface GenerateOptions {
  label?: string;
  hint?: string;
  errorMessage?: string;
  id?: string;
  name?: string;
  classes?: string;
  variant?: string;
  items?: Array<{
    text: string;
    value: string;
    hint?: string;
    checked?: boolean;
    selected?: boolean;
    conditional?: string;
  }>;
  [key: string]: unknown;
}
