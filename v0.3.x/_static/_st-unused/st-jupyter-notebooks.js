// // document.addEventListener("DOMContentLoaded", function() {
// //     const containers = document.querySelectorAll('.cell_output.docutils.container');
// //
// //     containers.forEach(container => {
// //         const toggleButton = document.createElement('button');
// //         toggleButton.innerHTML = '<span class="toggle-icon">+</span> Output';
// //         toggleButton.onclick = () => {
// //             container.classList.toggle('hidden');
// //             const icon = toggleButton.querySelector('.toggle-icon');
// //             icon.textContent = container.classList.contains('hidden') ? '+' : '-';
// //         };
// //         container.before(toggleButton);
// //         container.classList.add('hidden');
// //     });
// // });
// //
// // // CSS class to handle the hidden state
// // document.styleSheets[0].insertRule('.hidden { display: none; }', 0);
//
//
// // // THIS ONE
// // document.addEventListener("DOMContentLoaded", function () {
// //     const containers = document.querySelectorAll('.cell_output.docutils.container');
// //
// //     containers.forEach(container => {
// //         // Create a toggle button with an icon
// //         const toggleButton = document.createElement('button');
// //         toggleButton.innerHTML = '<span class="toggle-icon">+</span>'; // Initial icon
// //         toggleButton.className = 'toggle-button'; // Assign a class for styling
// //
// //         toggleButton.onclick = () => {
// //             container.classList.toggle('hidden');
// //             // Update the icon based on visibility
// //             const icon = toggleButton.querySelector('.toggle-icon');
// //             icon.textContent = container.classList.contains('hidden') ? '+' : '-';
// //         };
// //
// //         // Initially hide the content
// //         container.classList.add('hidden');
// //         container.before(toggleButton);
// //     });
// // });
//
// document.addEventListener("DOMContentLoaded", function() {
//     const containers = document.querySelectorAll('.cell_output.docutils.container');
//
//     containers.forEach(container => {
//         const toggleButton = document.createElement('button');
//         toggleButton.innerHTML = '<span class="toggle-icon">+</span>';
//         toggleButton.className = 'toggle-button';
//         toggleButton.setAttribute('data-tooltip', 'Click to see output'); // Set initial tooltip text
//
//         toggleButton.onclick = () => {
//             const isVisible = container.classList.toggle('hidden');
//             const icon = toggleButton.querySelector('.toggle-icon');
//             icon.textContent = isVisible ? '+': '-';
//             toggleButton.setAttribute('data-tooltip', isVisible ? 'Click to see output' : 'Click to collapse output');
//         };
//
//         container.classList.add('hidden');
//         container.before(toggleButton);
//     });
// });
