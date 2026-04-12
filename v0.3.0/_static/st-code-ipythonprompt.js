// Toggle for hiding IPython prompts (In [n]:, ...:) while keeping outputs visible.

(function () {
  const DEFAULT_HIDDEN = true;   // set to false to start with prompts shown
  const OFFSET_RIGHT_PX = 8;

  // Config flags (leave OUTPUT false to keep outputs visible)
  const HIDE_OUTPUT = false;      // .go (Generic.Output)
  const HIDE_TRACEBACKS = false;  // .gt (Generic.Traceback)
  const HIDE_BANNER = true;       // .gh (ipython header)

  function selectorParts() {
    const parts = ['.gp'];                  // prompts: In[n]: and ...:
    if (HIDE_BANNER) parts.push('.gh');     // ipython header
    if (HIDE_OUTPUT) parts.push('.go');     // outputs (usually keep false)
    if (HIDE_TRACEBACKS) parts.push('.gt'); // tracebacks
    return parts;
  }

  function* getHideable(root) {
    const parts = selectorParts().join(', ');
    if (parts) {
      for (const el of root.querySelectorAll(parts)) yield el;
    }

    // If hiding tracebacks, also wrap the bare text nodes after .gt so they can be hidden.
    if (HIDE_TRACEBACKS) {
      for (let el of root.querySelectorAll('.gt')) {
        let n = el;
        while ((n = n.nextSibling) && n.nodeType !== Node.DOCUMENT_NODE) {
          if (n.nodeType === Node.ELEMENT_NODE && n.matches(selectorParts().join(', '))) break;
          if (n.nodeType === Node.TEXT_NODE && n.textContent.trim()) {
            const span = document.createElement('span');
            n.after(span);
            span.appendChild(n);
            n = span;
          }
          if (n) yield n;
        }
      }
    }
  }

  function makeButton() {
    const btn = document.createElement('span');
    btn.classList.add('toggle-prompts-button');
    btn.textContent = '>>>';
    btn.title = 'Hide the prompts';
    btn.dataset.hidden = 'false';
    btn.style.cursor = 'pointer';
    btn.style.position = 'absolute';
    btn.style.top = '0';
    btn.style.right = OFFSET_RIGHT_PX + 'px';
    btn.style.fontFamily = 'monospace';
    btn.style.padding = '0 0.25em';
    btn.style.borderRadius = '0 3px 0 0';
    btn.style.userSelect = 'none';
    return btn;
  }

  function attachToBlock(block) {
    if (block.dataset.togglepromptsAttached === '1') return;

    // Only add a button if there are prompts (or other selected parts) in this block
    if (!block.querySelector(selectorParts().join(', '))) return;

    if (getComputedStyle(block).position === 'static') block.style.position = 'relative';

    const btn = makeButton();
    btn.addEventListener('click', evt => {
      evt.preventDefault();
      const hide = btn.dataset.hidden !== 'true';
      for (const node of getHideable(block)) node.hidden = hide;
      btn.dataset.hidden = hide ? 'true' : 'false';
      btn.title = hide ? 'Show the prompts' : 'Hide the prompts';
    });

    block.insertBefore(btn, block.firstChild);

    if (DEFAULT_HIDDEN) btn.click();

    block.dataset.togglepromptsAttached = '1';
  }

  function initAll() {
    const candidates = document.querySelectorAll(
      'div.highlight, pre.highlight, .highlight-ipython, .highlight-pycon, .highlight-python, .highlight-python3, .highlight-default'
    );
    candidates.forEach(attachToBlock);
  }

  const mo = new MutationObserver(initAll);
  mo.observe(document.documentElement, { childList: true, subtree: true });

  if (document.readyState !== 'loading') initAll();
  else document.addEventListener('DOMContentLoaded', initAll);
})();
