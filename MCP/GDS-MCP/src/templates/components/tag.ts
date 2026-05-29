import { GenerateOptions } from "../../knowledge/types.js";

export function renderTag(options: GenerateOptions): string {
  const text = options.label || (options.text as string) || "Tag";
  const colour = options.colour as string || options.variant || "";
  const colourClass = colour ? ` govuk-tag--${colour}` : "";

  return `<strong class="govuk-tag${colourClass}">
  ${text}
</strong>`;
}
