import { describe, it, expect } from "vitest";
import { handleGenerateMarkup } from "../../src/tools/generate-markup.js";

describe("generate_markup tool", () => {
  it("generates basic text input markup", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: { label: "Full name" },
    });

    const html = result.content[0].text;
    expect(html).toContain("govuk-form-group");
    expect(html).toContain("govuk-label");
    expect(html).toContain("govuk-input");
    expect(html).toContain("Full name");
    expect(html).toContain('for="full-name"');
    expect(html).toContain('id="full-name"');
  });

  it("generates text input with hint", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: {
        label: "Event name",
        hint: "The name you'll use on promotional material",
        id: "event-name",
      },
    });

    const html = result.content[0].text;
    expect(html).toContain("govuk-hint");
    expect(html).toContain("event-name-hint");
    expect(html).toContain('aria-describedby="event-name-hint"');
    expect(html).toContain("promotional material");
  });

  it("generates text input with error state", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: {
        label: "Full name",
        errorMessage: "Enter your full name",
        id: "full-name",
      },
    });

    const html = result.content[0].text;
    expect(html).toContain("govuk-form-group--error");
    expect(html).toContain("govuk-error-message");
    expect(html).toContain("govuk-visually-hidden");
    expect(html).toContain("Error:");
    expect(html).toContain("govuk-input--error");
    expect(html).toContain('aria-describedby="full-name-error"');
    expect(html).toContain("Enter your full name");
  });

  it("generates text input with hint and error (both in aria-describedby)", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: {
        label: "National Insurance number",
        hint: "It's on your National Insurance card",
        errorMessage: "Enter a National Insurance number",
        id: "ni-number",
      },
    });

    const html = result.content[0].text;
    expect(html).toContain('aria-describedby="ni-number-hint ni-number-error"');
  });

  it("generates text input with width class", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: {
        label: "Phone number",
        width: "10",
      },
    });

    const html = result.content[0].text;
    expect(html).toContain("govuk-input--width-10");
  });

  it("returns error for unknown component", () => {
    const result = handleGenerateMarkup({
      component: "nonexistent-widget",
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("not found");
  });

  it("includes accessibility notes when error state is used", () => {
    const result = handleGenerateMarkup({
      component: "text-input",
      options: {
        label: "Email",
        errorMessage: "Enter a valid email",
      },
    });

    expect(result.content[0].text).toContain("Accessibility notes");
    expect(result.content[0].text).toContain("visually hidden");
  });

  it("rejects unsafe URL protocol in options.href", () => {
    const result = handleGenerateMarkup({
      component: "back-link",
      options: {
        href: "javascript:alert('xss')",
      },
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("Unsafe URL protocol");
  });

  it("rejects unsafe URL protocol in items[].value (caught by post-render check)", () => {
    const result = handleGenerateMarkup({
      component: "breadcrumbs",
      options: {
        items: [
          { text: "Home", value: "javascript:alert('xss')" },
          { text: "Current", value: "/current" },
        ],
      },
    });

    // items[].value is not a URL_LIKE_OPTION_KEY (has dual semantics); unsafe
    // protocols in values that end up in href attributes are caught post-render.
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("unsafe href");
  });

  it("fieldset renders raw HTML content (content slot is not escaped)", () => {
    const result = handleGenerateMarkup({
      component: "fieldset",
      options: {
        label: "Personal details",
        content: '<div class="govuk-form-group"><label class="govuk-label" for="name">Full name</label><input class="govuk-input" id="name" name="name" type="text"></div>',
      },
    });

    expect(result.isError).toBeUndefined();
    const html = result.content[0].text;
    expect(html).toContain('class="govuk-input"');
    expect(html).toContain('for="name"');
    expect(html).not.toContain("&lt;input");
  });

  it("allows safe URLs with query parameters", () => {
    const result = handleGenerateMarkup({
      component: "back-link",
      options: {
        href: "/search?q=test&page=2",
      },
    });

    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("govuk-back-link");
  });

  it("allows safe absolute URLs", () => {
    const result = handleGenerateMarkup({
      component: "button",
      options: {
        variant: "start",
        href: "https://example.gov.uk/start",
        label: "Start now",
      },
    });

    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("govuk-button--start");
  });
});
