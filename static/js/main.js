$(document).ready(function () {
    // Assuming 'symptoms' is a global variable passed from your template, e.g., <script>var symptoms = {{ data|tojson }};</script>
    // If not, and it's hardcoded or loaded differently, ensure it's an array of strings.
    // If 'symptoms' is from Flask `data=json.dumps(data)`:
    // let symptomsData = JSON.parse('{{ data | safe }}'); // If using Flask template to pass data
    // For now, assuming `symptoms` is globally available and correctly parsed if needed.
    // If `symptoms` is the name of the variable in your HTML from Flask, make sure it's not conflicting.
    // Let's rename it here for clarity if it's loaded from the template as a string
    let availableSymptomsForSuggestions = [];
    if (typeof symptoms !== 'undefined') { // `symptoms` variable injected by Flask
        try {
            availableSymptomsForSuggestions = JSON.parse(symptoms);
        } catch (e) {
            console.error("Error parsing symptoms data from template:", e);
            // Fallback if parsing fails or symptoms isn't valid JSON
            availableSymptomsForSuggestions = ["fever", "cough", "headache"]; 
        }
    } else {
        console.warn("Global 'symptoms' variable for suggestions not found. Using fallback.");
        availableSymptomsForSuggestions = ["fever", "cough", "headache"]; // Fallback
    }


    let input = $("#message-text");
    let sendBtn = $("#send");
    let startOverBtn = $("#start-over");
    let symptomsList = $("#symptoms-list");
    let symptomsContainer = $("#symptoms-container");
    let chat = $("#conversation"); // Assuming this is your main chat messages container

    // Handler for input field changes for symptom suggestions
    input.on("input", function () {
        let insertedValue = $(this).val().toLowerCase().trim();
        symptomsList.empty();

        if (insertedValue.length > 1) {
            let suggestedSymptoms = $.fn.getSuggestedSymptoms(insertedValue);
            
            if (suggestedSymptoms.length === 0) {
                symptomsContainer.removeClass("show");
            } else {
                suggestedSymptoms.forEach(function(symptomText) {
                    let li = $("<li>").text(symptomText).on("click", function() {
                        input.val($(this).text()); // Set input to the clicked symptom
                        symptomsContainer.removeClass("show");
                        input.focus();
                    });
                    symptomsList.append(li);
                });
                symptomsContainer.addClass("show");
            }
        } else {
            symptomsContainer.removeClass("show");
        }
    });

    // Hide suggestions when clicking outside the input wrapper
    $(document).on("click", function(e) {
        if (!$(e.target).closest(".input-wrapper").length) { // Adjust '.input-wrapper' if necessary
            symptomsContainer.removeClass("show");
        }
    });

    // Start over functionality
    startOverBtn.on("click", function () {
        $.fn.startOver();
    });

    // Send button click handler
    sendBtn.on("click", function () {
        $.fn.handleUserMessage();
    });

    // Enter key handler in the input field
    input.on("keypress", function (e) {
        if (e.which === 13 && !e.shiftKey) { // Enter key pressed without Shift
            e.preventDefault(); // Prevent default form submission or newline
            $.fn.handleUserMessage();
        }
    });

    // Auto-resize input (useful if it's a textarea, less so for a single-line input)
    input.on("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // Main function to handle sending a user's message
    $.fn.handleUserMessage = function () {
        let messageText = input.val().trim();
        if (messageText) {
            symptomsContainer.removeClass("show"); // Hide suggestions
            $.fn.appendUserMessage(messageText); // Display user's message
            input.val(""); // Clear the input field
            input.css('height', 'auto'); // Reset height if auto-resizing
            
            $.fn.showTypingIndicator(); // Show bot typing indicator
            $.fn.getPredictedSymptom(messageText); // Send message to backend
            
            // Scroll to bottom of chat
            setTimeout(() => {
                $.fn.scrollToBottom();
            }, 100); // Short delay to ensure new message is rendered before scrolling
        }
    };

    // Function to start or restart the conversation
    $.fn.startOver = function () {
        chat.empty(); // Clear previous messages
        
        const welcomeMessageHTML = `
            <div class="message-wrapper bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i> 
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <p>Hello! I'm <strong>MedAssist</strong>, your AI medical assistant. I'm here to help you understand your symptoms better.</p>
                        <p>Please describe what symptoms you're currently experiencing. Be as detailed as possible - include when they started, their severity, and any patterns you've noticed.</p>
                        <p class="highlight">When you've described all your symptoms, simply type <strong>"Done"</strong> to get my analysis.</p>
                    </div>
                    <div class="message-time">${$.fn.getCurrentTime()}</div>
                </div>
            </div>
        `;
        
        chat.append(welcomeMessageHTML);
        input.val("");
        input.focus();
        
        // Optionally, send a "reset" or "done" with no symptoms to backend if needed to clear server-side state
        // For now, this seems to be handled by user_symptoms_identified_canonical.clear() in Flask's index route
        // If you need to explicitly tell the backend to reset symptom collection for the session:
        // $.fn.getPredictedSymptom("done", true); // This will clear user_symptoms_identified_canonical on backend due to "done"
    };

    // Function to append a user's message to the chat
    $.fn.appendUserMessage = function (text) {
        const userMessageHTML = `
            <div class="message-wrapper user-message">
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <p>${$.fn.escapeHtml(text)}</p> 
                    </div>
                    <div class="message-time">${$.fn.getCurrentTime()}</div>
                </div>
            </div>
        `;
        chat.append(userMessageHTML);
    };

    // Function to append a bot's message to the chat
    $.fn.appendBotMessage = function (htmlString) { // Expects an HTML string
        $(".typing-indicator").remove(); // Remove any existing typing indicator
        
        const botMessageHTML = `
            <div class="message-wrapper bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        ${htmlString} 
                    </div>
                    <div class="message-time">${$.fn.getCurrentTime()}</div>
                </div>
            </div>
        `;
        // Note: The <p> tags are removed from here because htmlString from Flask 
        // already contains <p>, <i>, <b> etc.
        // If response_text from Flask was plain text, you'd wrap it in <p>${htmlString}</p>
        chat.append(botMessageHTML);
        $.fn.scrollToBottom();
    };

    // Function to show a typing indicator for the bot
    $.fn.showTypingIndicator = function() {
        const typingIndicatorHTML = `
            <div class="message-wrapper bot-message typing-indicator">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        chat.append(typingIndicatorHTML);
        $.fn.scrollToBottom();
    };

    // Function to send text to backend and get a prediction/response
    $.fn.getPredictedSymptom = function (text, isReset = false) { // isReset is not currently used by backend logic based on text="done"
        const requestText = text; // For 'startOver', the Flask 'index' route already clears server-side symptoms.
                                  // If you send "done" from startOver, it might give "no symptoms" message if client-side is cleared first.
        
        $.ajax({
            url: "http://127.0.0.1:5000/symptom", // Ensure your Flask app runs on this URL
            data: JSON.stringify({ sentence: requestText }),
            contentType: "application/json; charset=utf-8",
            dataType: "json", // Expect JSON response
            type: "POST",
            success: function (response) { // 'response' is the parsed JSON object from Flask
                console.log("Bot AJAX response:", response);
                
                // Remove typing indicator *before* appending new message
                $(".typing-indicator").remove();

                if (response && response.response_text) {
                    // Add slight delay for more natural feel before showing the actual response
                    setTimeout(() => {
                        $.fn.appendBotMessage(response.response_text); // Pass the HTML string from response_text
                    }, 500 + Math.random() * 800); // 0.5 - 1.3 second delay
                } else if (response && response.error) {
                    console.error("Server returned an error:", response.error);
                    setTimeout(() => {
                        $.fn.appendBotMessage("An error occurred on the server: " + $.fn.escapeHtml(response.error));
                    }, 500);
                } else {
                    console.error("Unexpected response format from server:", response);
                    setTimeout(() => {
                        $.fn.appendBotMessage("Sorry, I received an unexpected response from the server.");
                    }, 500);
                }
            },
            error: function (xhr, status, error) {
                console.error("AJAX Error:", status, error, xhr.responseText);
                $(".typing-indicator").remove(); // Also remove indicator on error
                setTimeout(() => {
                    $.fn.appendBotMessage("I'm sorry, I'm having trouble connecting right now. Please try again in a moment.");
                }, 500);
            }
        });
    };

    // Function to get suggested symptoms based on user input
    $.fn.getSuggestedSymptoms = function (val) {
        let Sugsymptom = [];
        if (availableSymptomsForSuggestions && availableSymptomsForSuggestions.length > 0) {
            $.each(availableSymptomsForSuggestions, function (i, symptomText) {
                if (symptomText.toLowerCase().includes(val.toLowerCase())) {
                    Sugsymptom.push(symptomText);
                }
            });
        }
        return Sugsymptom.slice(0, 5); // Show top 5 suggestions
    };

    // Utility function to get current time as HH:MM AM/PM
    $.fn.getCurrentTime = function() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    // Utility function to escape HTML special characters to prevent XSS
    $.fn.escapeHtml = function(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    // Utility function to scroll the chat to the bottom
    $.fn.scrollToBottom = function() {
        // Using jQuery animate for a smoother scroll
        chat.stop().animate({
            scrollTop: chat[0].scrollHeight
        }, 300);
    };

    // Initialize the chat application
    $.fn.startOver(); // Display initial welcome message
    input.focus();
});

// Add CSS for typing indicator animation (moved here for better organization)
// This ensures it's added after the DOM is ready.
$(document).ready(function() {
    const typingCSS = `
    <style>
    .typing-dots {
        display: flex;
        align-items: center; /* Vertically align dots if bubble has padding */
        gap: 5px; /* Space between dots */
        height: 100%; /* Ensure it fills bubble if needed */
    }

    .typing-dots span {
        width: 8px;
        height: 8px;
        background-color: #94a3b8; /* Dot color */
        border-radius: 50%;
        opacity: 0; /* Start invisible */
        animation: typingAnimation 1.4s infinite ease-in-out;
    }

    .typing-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }

    .typing-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0s;
    }

    @keyframes typingAnimation {
        0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }

    .typing-indicator .message-bubble { /* Style the bubble containing the dots */
        padding: 10px 12px; /* Adjust padding as needed */
        display: inline-block; /* Make bubble only as wide as content */
    }
    </style>
    `;
    $("head").append(typingCSS);
});