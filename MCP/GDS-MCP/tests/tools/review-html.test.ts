import { describe, it, expect } from "vitest";
import { handleReviewHtml } from "../../src/tools/review-html.js";

describe("review_html tool", () => {
  it("passes valid text input HTML", () => {
    const html = `
<div class="govuk-form-group">
  <label class="govuk-label" for="name">Full name</label>
  <input class="govuk-input" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("Passed checks");
    expect(text).not.toContain("Errors (must fix)");
  });

  it("flags input without label", () => {
    const html = `
<div class="govuk-form-group">
  <input class="govuk-input" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("Errors (must fix)");
    expect(text).toContain("label");
  });

  it("flags placeholder-only input", () => {
    const html = `<input id="search" placeholder="Search" type="text">`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("placeholder");
  });

  it("flags error message without visually hidden prefix", () => {
    const html = `
<div class="govuk-form-group govuk-form-group--error">
  <label class="govuk-label" for="name">Name</label>
  <p class="govuk-error-message">Enter your name</p>
  <input class="govuk-input govuk-input--error" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("visually hidden");
  });

  it("passes error message with correct format", () => {
    const html = `
<div class="govuk-form-group govuk-form-group--error">
  <label class="govuk-label" for="name">Name</label>
  <p class="govuk-error-message">
    <span class="govuk-visually-hidden">Error:</span> Enter your name
  </p>
  <input class="govuk-input govuk-input--error" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).not.toContain("Errors (must fix)");
  });

  it("flags missing GOV.UK classes on form elements", () => {
    const html = `
<div>
  <label for="name">Name</label>
  <input id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("govuk-");
  });

  it("runs component-specific rules when component slug provided", () => {
    const html = `
<div class="govuk-form-group">
  <label class="govuk-label" for="name">Name</label>
  <input class="govuk-input" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html, component: "text-input" });
    const text = result.content[0].text;

    expect(text).toContain("Passed checks");
  });

  it("respects lenient strictness (only errors)", () => {
    const html = `
<div>
  <label for="name">Name</label>
  <input class="govuk-input" id="name" name="name" type="text">
</div>`;

    const result = handleReviewHtml({ html, strictness: "lenient" });
    const text = result.content[0].text;

    // Should not flag missing govuk-form-group (warning) in lenient mode
    expect(text).not.toContain("govuk-form-group");
  });

  it("reports all checks passed for well-formed GOV.UK HTML", () => {
    const html = `
<div class="govuk-form-group">
  <label class="govuk-label" for="email">Email address</label>
  <div id="email-hint" class="govuk-hint">We'll use this to send your confirmation</div>
  <input class="govuk-input" id="email" name="email" type="email" aria-describedby="email-hint">
</div>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("Passed checks");
  });

  it("flags unsafe URL protocols in href/src/action attributes", () => {
    const html = `<a class="govuk-link" href="javascript:alert('xss')">Bad link</a>`;

    const result = handleReviewHtml({ html });
    const text = result.content[0].text;

    expect(text).toContain("Errors (must fix)");
    expect(text).toContain("unsafe");
    expect(text).toContain("protocol");
  });
});
