import { GenerateOptions } from "../../knowledge/types.js";

export function renderButton(options: GenerateOptions): string {
  const text = options.label || (options.text as string) || "Submit";
  const type = (options.type as string) || "submit";
  const variant = options.variant || "";
  const isDisabled = !!options.disabled;

  let variantClass = "";
  if (variant === "secondary") variantClass = " govuk-button--secondary";
  else if (variant === "warning") variantClass = " govuk-button--warning";
  else if (variant === "start") variantClass = " govuk-button--start";
  else if (variant === "inverse") variantClass = " govuk-button--inverse";

  const disabledAttrs = isDisabled ? " disabled aria-disabled=\"true\"" : "";

  if (variant === "start") {
    return `<a href="${(options.href as string) || "/"}" role="button" draggable="false" class="govuk-button govuk-button--start" data-module="govuk-button">
  ${text}
  <svg class="govuk-button__start-icon" xmlns="http://www.w3.org/2000/svg" width="17.5" height="19" viewBox="0 0 33 40" aria-hidden="true" focusable="false">
    <path fill="currentColor" d="M0 0h13l20 20-20 20H0l20-20z" />
  </svg>
</a>`;
  }

  return `<button type="${type}" class="govuk-button${variantClass}" data-module="govuk-button"${disabledAttrs}>
  ${text}
</button>`;
}
