import { GenerateOptions } from "../../knowledge/types.js";

/**
 * The `content` option accepts raw HTML (slot type: "html") and is NOT
 * HTML-escaped. It should contain pre-authored, trusted form field markup
 * such as a govuk-form-group wrapping a text input.
 * `sanitizeGenerateOptions` skips escaping for the `content` key specifically.
 */
export function renderFieldset(options: GenerateOptions): string {
  const legend = options.label || (options.legend as string) || "Legend";
  const content = (options.content as string) || "<!-- Form fields go here -->";

  return `<fieldset class="govuk-fieldset">
  <legend class="govuk-fieldset__legend">
    ${legend}
  </legend>
  ${content}
</fieldset>`;
}
