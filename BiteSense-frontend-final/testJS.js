function file_explorer() {
    document.getElementById('selectfile').click(); // Trigger the file input click
}

function upload_file(event) {
    event.preventDefault(); // Prevent the default behavior (like opening the file)
    const files = event.dataTransfer.files; // Get the files from the event
    handleFiles(files); // Call the function to handle the files
}

function drag_over() {
    const dropZone = document.getElementById('drop_file_zone');
    dropZone.classList.add('dragging'); // Add a class to indicate dragging
}

function drag_leave() {
    const dropZone = document.getElementById('drop_file_zone');
    dropZone.classList.remove('dragging'); // Remove the class when leaving the zone
}

document.getElementById('selectfile').addEventListener('change', function() {
    const files = this.files; // Get the selected files
    handleFiles(files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0]; // Get the first file
        const reader = new FileReader();

        reader.onload = function(e) {
            console.log('File uploaded:', file.name); // You can add code here to display the uploaded image
        };

        reader.readAsDataURL(file); // Read the file as a data URL
    }
}
