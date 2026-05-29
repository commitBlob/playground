import { describe, it, expect } from "vitest";
import { handleSuggestComponent } from "../../src/tools/suggest-component.js";

describe("suggest_component tool", () => {
  it("suggests text-input for entering a name", () => {
    const result = handleSuggestComponent({
      useCase: "users need to enter their name",
    });

    expect(result.content[0].type).toBe("text");
    expect(result.content[0].text).toContain("Text input");
    expect(result.content[0].text).toContain("Recommended");
  });

  it("suggests text-input for short text entry", () => {
    const result = handleSuggestComponent({
      useCase: "single line text field for a reference number",
    });

    expect(result.content[0].text).toContain("Text input");
  });

  it("includes quick-start HTML", () => {
    const result = handleSuggestComponent({
      useCase: "enter their name",
    });

    expect(result.content[0].text).toContain("```html");
    expect(result.content[0].text).toContain("govuk-input");
    expect(result.content[0].text).toContain("govuk-label");
  });

  it("includes anti-use-cases", () => {
    const result = handleSuggestComponent({
      useCase: "enter their name",
    });

    expect(result.content[0].text).toContain("When NOT to use");
    expect(result.content[0].text).toContain("Textarea");
  });

  it("returns no-match message for unrecognised use case", () => {
    const result = handleSuggestComponent({
      useCase: "quantum flux capacitor integration",
    });

    expect(result.content[0].text).toContain("No GOV.UK Design System component found");
  });

  it("includes related components", () => {
    const result = handleSuggestComponent({
      useCase: "enter their name",
    });

    expect(result.content[0].text).toContain("Related components");
    expect(result.content[0].text).toContain("textarea");
  });
});
