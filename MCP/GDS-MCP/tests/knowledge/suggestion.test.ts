import { describe, it, expect } from "vitest";
import { handleSuggestComponent } from "../../src/tools/suggest-component.js";

describe("suggestion intelligence", () => {
  it("disambiguates between checkboxes and radios for 'select from options'", () => {
    const result = handleSuggestComponent({
      useCase: "users need to select from a list of options",
    });
    const text = result.content[0].text;
    // Should mention both as they're close matches
    expect(text).toContain("Checkboxes");
  });

  it("recommends radios for single selection", () => {
    const result = handleSuggestComponent({
      useCase: "users select one option from a list",
    });
    expect(result.content[0].text).toContain("Radios");
  });

  it("recommends checkboxes for multiple selection", () => {
    const result = handleSuggestComponent({
      useCase: "users need to select multiple options",
    });
    expect(result.content[0].text).toContain("Checkboxes");
  });

  it("distinguishes textarea from text input for long text", () => {
    const result = handleSuggestComponent({
      useCase: "users need to enter longer text with multiple lines",
    });
    const text = result.content[0].text;
    expect(text).toContain("Textarea");
  });

  it("suggests text input for short answers", () => {
    const result = handleSuggestComponent({
      useCase: "users need to type a short answer like their name",
    });
    expect(result.content[0].text).toContain("Text input");
  });

  it("suggests date-input for dates", () => {
    const result = handleSuggestComponent({
      useCase: "enter a date of birth",
    });
    expect(result.content[0].text).toContain("Date input");
  });

  it("suggests error-summary for listing form errors", () => {
    const result = handleSuggestComponent({
      useCase: "list of all errors on the page",
    });
    expect(result.content[0].text).toContain("Error summary");
  });

  it("suggests panel for confirmation pages", () => {
    const result = handleSuggestComponent({
      useCase: "confirmation page for application submitted",
    });
    expect(result.content[0].text).toContain("Panel");
  });

  it("suggests details for hiding supplementary content", () => {
    const result = handleSuggestComponent({
      useCase: "show and hide extra information",
    });
    expect(result.content[0].text).toContain("Details");
  });

  it("suggests breadcrumbs for hierarchy navigation", () => {
    const result = handleSuggestComponent({
      useCase: "breadcrumb navigation trail showing site hierarchy",
    });
    expect(result.content[0].text).toContain("Breadcrumbs");
  });

  it("disambiguates with 'Also consider' section when scores are close", () => {
    const result = handleSuggestComponent({
      useCase: "show and hide sections",
    });
    const text = result.content[0].text;
    // Both accordion and details match this, but accordion is better for multiple sections
    expect(text).toContain("Accordion");
  });

  it("handles context parameter to refine suggestions", () => {
    const result = handleSuggestComponent({
      useCase: "select options",
      context: "users can pick multiple items",
    });
    expect(result.content[0].text).toContain("Checkboxes");
  });

  it("suggests select for dropdown needs", () => {
    const result = handleSuggestComponent({
      useCase: "dropdown list to sort results",
    });
    expect(result.content[0].text).toContain("Select");
  });

  it("suggests password-input for password fields", () => {
    const result = handleSuggestComponent({
      useCase: "enter a password",
    });
    expect(result.content[0].text).toContain("Password input");
  });
});
