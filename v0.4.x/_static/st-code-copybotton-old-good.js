// // Copy ONLY IPython inputs (no prompts, no outputs), cross-browser.
// // Works with Sphinx/Material + Pygments markup.
//
// (function () {
//   function isEl(n, cls) {
//     return n && n.nodeType === Node.ELEMENT_NODE && n.classList.contains(cls);
//   }
//   function isPrompt(n) { return isEl(n, 'gp'); }   // Generic.Prompt
//   function isOutput(n) { return isEl(n, 'go'); }   // Generic.Output
//   function isIpythonHeader(n) { return isEl(n, 'gh'); } // ipython header
//   function isTraceback(n) { return isEl(n, 'gt'); } // Generic.Traceback
//
//   function isInputPromptText(t) {
//     const s = (t || '').trim();
//     return /^In\s*\[\d+\]:$/.test(s) || s === '...:'; // input prompt or continuation
//   }
//
//   // Turn a code container's childNodes into "visual" lines.
//   // Splits on <br> AND on embedded '\n' inside text nodes.
//   function getLogicalLines(codeEl) {
//     const lines = [];
//     let current = [];
//
//     const flush = () => { lines.push(current); current = []; };
//
//     function pushNodeOrSplit(n) {
//       if (n.nodeType === Node.TEXT_NODE) {
//         const parts = (n.textContent || '').split(/\r?\n/);
//         for (let i = 0; i < parts.length; i++) {
//           if (i > 0) flush();
//           if (parts[i].length) {
//             // Re-wrap each text part so we can keep positions per "line"
//             const t = document.createTextNode(parts[i]);
//             current.push(t);
//           }
//         }
//       } else if (n.nodeName === 'BR') {
//         flush();
//       } else {
//         current.push(n);
//       }
//     }
//
//     // Some themes use <pre><code>..., others <pre> directly
//     const container = codeEl.querySelector('pre > code') || codeEl.querySelector('pre') || codeEl;
//
//     container.childNodes.forEach(pushNodeOrSplit);
//     flush(); // last line
//     // remove possible empty trailing line
//     if (lines.length && lines[lines.length - 1].length === 0) lines.pop();
//     return lines;
//   }
//
//   function extractInputOnly(blockEl) {
//     // The outer "code area" wrapper (theme dependent)
//     const codeEl = blockEl;
//
//     // Build logical lines
//     const lines = getLogicalLines(codeEl);
//
//     const out = [];
//
//     lines.forEach(nodes => {
//       if (!nodes || !nodes.length) return;
//
//       // If the line contains any OUTPUT token, skip it entirely
//       if (nodes.some(isOutput)) return;
//
//       // Find first prompt token on the line (if any)
//       const pIdx = nodes.findIndex(isPrompt);
//       if (pIdx === -1) return; // no prompt → not input → skip
//
//       const pText = nodes[pIdx].textContent || '';
//       if (!isInputPromptText(pText)) return; // Out[n]: or other → skip
//
//       // Collect the rest of this same visual line AFTER the prompt
//       const tail = nodes.slice(pIdx + 1)
//         .filter(n => !isIpythonHeader(n) && !isTraceback(n) && !isOutput(n))
//         .map(n => n.textContent || '')
//         .join('');
//
//       // Remove a single leading space that often follows the prompt
//       out.push(tail.replace(/^\s/, ''));
//     });
//
//     return out.join('\n').trimEnd();
//   }
//
//   async function copyText(text) {
//     try {
//       if (navigator.clipboard?.writeText) {
//         await navigator.clipboard.writeText(text);
//       } else {
//         // Safari / legacy fallback
//         const ta = document.createElement('textarea');
//         ta.value = text;
//         ta.setAttribute('readonly', '');
//         ta.style.position = 'fixed';
//         ta.style.top = '-9999px';
//         document.body.appendChild(ta);
//         ta.select();
//         document.execCommand('copy');
//         document.body.removeChild(ta);
//       }
//     } catch (e) {
//       console.error('Copy failed, showing prompt fallback:', e);
//       window.prompt('Copy the code below:', text);
//     }
//   }
//
//   function handleClick(ev) {
//     ev.preventDefault();
//     const btn = ev.currentTarget;
//     const targetSel = btn.getAttribute('data-clipboard-target');
//     const target = document.querySelector(targetSel);
//     if (!target) return;
//
//     const codeOnly = extractInputOnly(target);
//     copyText(codeOnly);
//   }
//
//   function attach() {
//     document.querySelectorAll('.md-clipboard.md-icon[data-clipboard-target]')
//       .forEach(btn => {
//         if (btn.dataset.ipyInputOnly === '1') return;
//         // Capture phase to override theme’s default copy handler
//         btn.addEventListener('click', handleClick, { capture: true });
//         btn.dataset.ipyInputOnly = '1';
//       });
//   }
//
//   if (document.readyState !== 'loading') attach();
//   else document.addEventListener('DOMContentLoaded', attach);
//
//   // Handle dynamic inserts (tabs, search, etc.)
//   new MutationObserver(attach).observe(document.documentElement, { childList: true, subtree: true });
// })();
