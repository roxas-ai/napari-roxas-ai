window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("script[type='math/tex']").forEach(node => {
    const math = node.textContent;
    node.outerHTML = `\\(${math}\\)`;
  });
});
