import { ComponentRule, RuleResult } from "../types.js";

export const designPatternRules: ComponentRule[] = [
  {
    id: "govuk-classes-present",
    severity: "warning",
    description: "Elements should use GOV.UK Frontend CSS classes (govuk-*)",
    check: (html: string): RuleResult => {
      const hasFormElements =
        /<(input|select|textarea|button)[^>]*>/.test(html);
      if (!hasFormElements) {
        return { passed: true, message: "No form elements to check." };
      }
      const hasGovukClasses = /class=["'][^"']*govuk-/.test(html);
      if (!hasGovukClasses) {
        return {
          passed: false,
          message:
            "Form elements found but no GOV.UK Frontend classes (govuk-*) detected.",
          suggestion:
            "Use GOV.UK Frontend classes like govuk-input, govuk-button, govuk-select for consistent styling.",
        };
      }
      return { passed: true, message: "GOV.UK Frontend classes are in use." };
    },
  },
  {
    id: "form-group-wrapper",
    severity: "warning",
    description: "Form inputs should be wrapped in a govuk-form-group div",
    check: (html: string): RuleResult => {
      const hasInput = /<input[^>]*class=["'][^"']*govuk-input/.test(html);
      if (!hasInput) {
        return { passed: true, message: "No GOV.UK inputs to check." };
      }
      const hasFormGroup = /govuk-form-group/.test(html);
      if (!hasFormGroup) {
        return {
          passed: false,
          message: "GOV.UK input found without a wrapping govuk-form-group container.",
          suggestion:
            'Wrap the label + input in <div class="govuk-form-group">...</div>.',
        };
      }
      return { passed: true, message: "Inputs are wrapped in govuk-form-group." };
    },
  },
  {
    id: "button-data-module",
    severity: "warning",
    description: "GOV.UK buttons should have data-module=\"govuk-button\"",
    check: (html: string): RuleResult => {
      const hasButton = /class=["'][^"']*govuk-button/.test(html);
      if (!hasButton) {
        return { passed: true, message: "No GOV.UK buttons present." };
      }
      const hasDataModule = /data-module=["']govuk-button["']/.test(html);
      if (!hasDataModule) {
        return {
          passed: false,
          message:
            "GOV.UK button is missing data-module=\"govuk-button\" for double-click prevention.",
          suggestion:
            'Add data-module="govuk-button" to the button element.',
        };
      }
      return {
        passed: true,
        message: "Button has data-module attribute.",
      };
    },
  },
  {
    id: "fieldset-for-grouped-inputs",
    severity: "warning",
    description:
      "Related checkboxes or radios should be wrapped in a fieldset with legend",
    check: (html: string): RuleResult => {
      const hasMultipleCheckboxes =
        (html.match(/type=["']checkbox["']/g) || []).length > 1;
      const hasMultipleRadios =
        (html.match(/type=["']radio["']/g) || []).length > 1;
      if (!hasMultipleCheckboxes && !hasMultipleRadios) {
        return { passed: true, message: "No grouped inputs to check." };
      }
      const hasFieldset = /<fieldset[^>]*>/.test(html);
      const hasLegend = /<legend[^>]*>/.test(html);
      if (!hasFieldset || !hasLegend) {
        return {
          passed: false,
          message:
            "Multiple checkboxes/radios found without a wrapping <fieldset> and <legend>.",
          suggestion:
            "Wrap grouped inputs in a <fieldset> with a <legend> describing the group.",
        };
      }
      return {
        passed: true,
        message: "Grouped inputs are wrapped in fieldset with legend.",
      };
    },
  },
];
