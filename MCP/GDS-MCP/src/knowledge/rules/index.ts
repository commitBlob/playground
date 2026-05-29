import { ComponentRule, RuleResult } from "../types.js";
import { accessibilityRules } from "./accessibility.js";
import { designPatternRules } from "./design-patterns.js";

const allRules: ComponentRule[] = [...accessibilityRules, ...designPatternRules];

export interface ReviewResult {
  ruleId: string;
  severity: "error" | "warning";
  description: string;
  result: RuleResult;
}

export function runAllRules(
  html: string,
  strictness: "strict" | "moderate" | "lenient" = "moderate"
): ReviewResult[] {
  const results: ReviewResult[] = [];

  for (const rule of allRules) {
    if (strictness === "lenient" && rule.severity === "warning") continue;

    const result = rule.check(html);
    results.push({
      ruleId: rule.id,
      severity: rule.severity,
      description: rule.description,
      result,
    });
  }

  return results;
}

export function runComponentRules(
  html: string,
  componentRules: ComponentRule[]
): ReviewResult[] {
  return componentRules.map((rule) => ({
    ruleId: rule.id,
    severity: rule.severity,
    description: rule.description,
    result: rule.check(html),
  }));
}
