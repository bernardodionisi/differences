// document.addEventListener('DOMContentLoaded', function() {
//     const ipythonDiv = document.querySelector('.highlight-ipython');
//     const highlightDiv = ipythonDiv.querySelector('.highlight');
//     const preElement = highlightDiv.querySelector('pre');
//
//     // Create and configure the toggle button
//     const toggleButton = document.createElement('button');
//     toggleButton.textContent = 'Hide';
//     toggleButton.className = 'toggle-button'; // Ensuring class is set correctly for CSS targeting
//
//     // Append the toggle button to the highlight-ipython div, not the highlight div
//     ipythonDiv.insertBefore(toggleButton, ipythonDiv.firstChild); // Places it before anything else in the ipythonDiv
//
//     // Event listener for the toggle button
//     toggleButton.addEventListener('click', function() {
//         if (preElement.style.display !== 'none') {
//             preElement.style.display = 'none'; // Hide the content
//             toggleButton.textContent = 'Show';
//         } else {
//             preElement.style.display = ''; // Show the content
//             toggleButton.textContent = 'Hide';
//         }
//     });
// });
//

// REMOVE BOTTOM NAVIGATION
document.addEventListener('DOMContentLoaded', (event) => {
    const footer = document.querySelector('.md-footer__inner.md-grid');
    if (footer) {
        footer.remove(); // Removes the element from the DOM
    }
});
