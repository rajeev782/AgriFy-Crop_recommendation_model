$(document).ready(function() {
    $("#cropForm").submit(function(event) {
        event.preventDefault();

        // Show loading spinner
        $("#loading").show();
        $("#result").hide();

        let formData = {
            N: $("#N").val(),
            P: $("#P").val(),
            K: $("#K").val(),
            temperature: $("#temperature").val(),
            humidity: $("#humidity").val(),
            ph: $("#ph").val(),
            rainfall: $("#rainfall").val()
        };

        console.log("Sending Data:", formData);  // Debugging

        $.ajax({
            type: "POST",
            url: "http://127.0.0.1:5001/predict",  // Ensure this matches Flask route
            contentType: "application/json",
            data: JSON.stringify(formData),
            success: function(response) {
                console.log("Received Response:", response);
                if (response.crop) {
                    $("#crop-name").text(response.crop);
                    $("#loading").hide();
                    $("#result").fadeIn();
                } else {
                    alert("Error: No crop prediction received.");
                }
            },
            error: function(xhr, status, error) {
                console.error("AJAX Error:", xhr.responseText);
                alert("Error making prediction. Check the console for details.");
                $("#loading").hide();
            }
        });
    });
});
