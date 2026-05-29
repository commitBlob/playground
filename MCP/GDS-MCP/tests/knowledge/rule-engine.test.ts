import { describe, it, expect } from "vitest";
import { handleReviewHtml } from "../../src/tools/review-html.js";

describe("action link context rule", () => {
  it("flags bare 'Change' link", () => {
    const html = `<a class="govuk-link" href="/change">Change</a>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("context");
  });

  it("passes Change link with visually hidden context", () => {
    const html = `<a class="govuk-link" href="/change">Change<span class="govuk-visually-hidden"> name</span></a>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).not.toContain("lacks context");
  });

  it("flags bare 'Remove' link", () => {
    const html = `<a href="/remove">Remove</a>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("context");
  });
});

describe("aria-describedby on error rule", () => {
  it("flags input with error but no aria-describedby", () => {
    const html = `
<div class="govuk-form-group govuk-form-group--error">
  <label class="govuk-label" for="name">Name</label>
  <p class="govuk-error-message"><span class="govuk-visually-hidden">Error:</span> Enter name</p>
  <input class="govuk-input" id="name" name="name" type="text">
</div>`;
    const result = handleReviewHtml({ html, strictness: "strict" });
    expect(result.content[0].text).toContain("aria-describedby");
  });

  it("passes input with error and aria-describedby", () => {
    const html = `
<div class="govuk-form-group govuk-form-group--error">
  <label class="govuk-label" for="name">Name</label>
  <p id="name-error" class="govuk-error-message"><span class="govuk-visually-hidden">Error:</span> Enter name</p>
  <input class="govuk-input" id="name" name="name" type="text" aria-describedby="name-error">
</div>`;
    const result = handleReviewHtml({ html });
    const text = result.content[0].text;
    expect(text).not.toContain("input lacks aria-describedby");
  });
});

describe("fieldset for grouped inputs rule", () => {
  it("flags multiple radios without fieldset", () => {
    const html = `
<div>
  <input type="radio" id="r1" name="q" value="a"><label for="r1">A</label>
  <input type="radio" id="r2" name="q" value="b"><label for="r2">B</label>
</div>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("fieldset");
  });

  it("passes grouped radios with fieldset", () => {
    const html = `
<fieldset class="govuk-fieldset">
  <legend class="govuk-fieldset__legend">Question</legend>
  <div class="govuk-radios">
    <input type="radio" id="r1" name="q" value="a"><label for="r1">A</label>
    <input type="radio" id="r2" name="q" value="b"><label for="r2">B</label>
  </div>
</fieldset>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).not.toContain("Errors (must fix)");
  });
});

describe("table header scope rule", () => {
  it("flags th without scope", () => {
    const html = `
<table class="govuk-table">
  <thead><tr><th>Header</th></tr></thead>
  <tbody><tr><td>Data</td></tr></tbody>
</table>`;
    const result = handleReviewHtml({ html, strictness: "strict" });
    expect(result.content[0].text).toContain("scope");
  });

  it("passes th with scope", () => {
    const html = `
<table class="govuk-table">
  <thead><tr><th scope="col">Header</th></tr></thead>
  <tbody><tr><td>Data</td></tr></tbody>
</table>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("Passed");
  });
});

describe("textarea label rule", () => {
  it("flags textarea without label", () => {
    const html = `<textarea class="govuk-textarea" id="comments" name="comments"></textarea>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("label");
  });
});

describe("select label rule", () => {
  it("flags select without label", () => {
    const html = `<select class="govuk-select" id="sort" name="sort"><option>A</option></select>`;
    const result = handleReviewHtml({ html });
    expect(result.content[0].text).toContain("label");
  });
});

describe("error summary linking rule", () => {
  it("flags error summary without field links", () => {
    const html = `
<div class="govuk-error-summary" data-module="govuk-error-summary">
  <div role="alert">
    <h2 class="govuk-error-summary__title">There is a problem</h2>
    <ul class="govuk-list govuk-error-summary__list">
      <li>Enter your name</li>
    </ul>
  </div>
</div>`;
    const result = handleReviewHtml({ html, component: "error-summary" });
    expect(result.content[0].text).toContain("link");
  });
});
