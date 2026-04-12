/* Handles copy for both ipython directives (input-only) and plain code blocks,
   plus a minimal “Copied” toast.
*/
(function () {
    // ---------- Helpers: token checks ----------
    function isEl(n, cls) {
        return n && n.nodeType === Node.ELEMENT_NODE && n.classList.contains(cls);
    }

    function isPrompt(n) {
        return isEl(n, 'gp');
    }   // Generic.Prompt (ipython)
    function isOutput(n) {
        return isEl(n, 'go');
    }   // Generic.Output
    function isIpythonHeader(n) {
        return isEl(n, 'gh');
    }

    function isTraceback(n) {
        return isEl(n, 'gt');
    }

    function isInputPromptText(t) {
        const s = (t || '').trim();
        return /^In\s*\[\d+\]:$/.test(s) || s === '...:'; // ipython input or continuation
    }

    function getCodeContainer(blockEl) {
        return (
            blockEl.querySelector('pre > code') ||
            blockEl.querySelector('pre') ||
            blockEl
        );
    }

    function hasIpyPrompts(blockEl) {
        const c = getCodeContainer(blockEl);
        return !!c.querySelector('.gp');
    }

    // ---------- Logical lines from DOM ----------
    function getLogicalLines(codeEl) {
        const container = getCodeContainer(codeEl);
        const lines = [];
        let current = [];
        const flush = () => {
            lines.push(current);
            current = [];
        };

        function pushNodeOrSplit(n) {
            if (n.nodeType === Node.TEXT_NODE) {
                const parts = (n.textContent || '').split(/\r?\n/);
                for (let i = 0; i < parts.length; i++) {
                    if (i > 0) flush();
                    if (parts[i].length) current.push(document.createTextNode(parts[i]));
                }
            } else if (n.nodeName === 'BR') {
                flush();
            } else {
                current.push(n);
            }
        }

        container.childNodes.forEach(pushNodeOrSplit);
        flush();
        if (lines.length && lines[lines.length - 1].length === 0) lines.pop();
        return lines;
    }

    // ---------- Extraction strategies ----------
    function extractIpyInputOnly(blockEl) {
        const lines = getLogicalLines(blockEl);
        const out = [];

        lines.forEach(nodes => {
            if (!nodes || !nodes.length) return;
            if (nodes.some(isOutput)) return;

            const pIdx = nodes.findIndex(isPrompt);
            if (pIdx === -1) return;

            const pText = nodes[pIdx].textContent || '';
            if (!isInputPromptText(pText)) return;

            const tail = nodes
                .slice(pIdx + 1)
                .filter(n => !isIpythonHeader(n) && !isTraceback(n) && !isOutput(n))
                .map(n => n.textContent || '')
                .join('');

            out.push(tail.replace(/^\s/, ''));
        });

        return out.join('\n').trimEnd();
    }

    function extractAllCode(blockEl) {
        const clone = blockEl.cloneNode(true);
        // Remove common line-number gutters
        clone.querySelectorAll('.linenos, .linenodiv, td.linenos, .line-numbers-rows').forEach(el => el.remove());
        clone.querySelectorAll('table.highlighttable td.linenos').forEach(el => el.remove());

        const container =
            clone.querySelector('pre > code') ||
            clone.querySelector('pre') ||
            clone;

        return (container.textContent || '').replace(/\s+$/, '');
    }

    // ---------- Toast / feedback ----------
    let _toast, _revertTimer;

    function getToast() {
        if (_toast) return _toast;
        _toast = document.createElement('div');
        _toast.className = 'copy-toast';
        _toast.setAttribute('role', 'status');
        _toast.setAttribute('aria-live', 'polite');
        document.body.appendChild(_toast);
        return _toast;
    }

    function announceCopied(msg = 'Copied') {
        const t = getToast();
        t.textContent = msg;
        // retrigger transition
        // eslint-disable-next-line no-unused-expressions
        t.offsetHeight;
        t.classList.add('show');
        clearTimeout(_revertTimer);
        _revertTimer = setTimeout(() => t.classList.remove('show'), 1200);
    }

    function feedbackCopied(btn, msg = 'Copied') {
        announceCopied(msg);
        if (!btn) return;
        const prev = btn.getAttribute('aria-label') || btn.title || '';
        btn.setAttribute('aria-label', msg);
        btn.title = msg;
        btn.classList.add('copied');
        setTimeout(() => {
            if (prev) btn.setAttribute('aria-label', prev); else btn.removeAttribute('aria-label');
            btn.title = prev;
            btn.classList.remove('copied');
        }, 1200);
    }

    // ---------- Copy + attach ----------
    async function copyText(text, btn) {
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.top = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
            feedbackCopied(btn);
        } catch (e) {
            console.error('Copy failed, showing prompt fallback:', e);
            window.prompt('Copy the code below:', text);
        }
    }

    function handleClick(ev) {
        ev.preventDefault();
        ev.stopPropagation();
        if (ev.stopImmediatePropagation) ev.stopImmediatePropagation();

        const btn = ev.currentTarget;
        const targetSel = btn.getAttribute('data-clipboard-target');
        const target = document.querySelector(targetSel);
        if (!target) return;

        const text = hasIpyPrompts(target) ? extractIpyInputOnly(target) : extractAllCode(target);
        copyText(text, btn);
    }

    function attach() {
        document.querySelectorAll('.md-clipboard.md-icon[data-clipboard-target]').forEach(btn => {
            if (btn.dataset.ipyInputOnly === '1') return;
            btn.addEventListener('click', handleClick, {capture: true});
            btn.dataset.ipyInputOnly = '1';
        });
    }

    if (document.readyState !== 'loading') attach();
    else document.addEventListener('DOMContentLoaded', attach);

    new MutationObserver(attach).observe(document.documentElement, {
        childList: true,
        subtree: true
    });
})();
