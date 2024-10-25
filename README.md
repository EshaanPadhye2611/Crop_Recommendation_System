The Crop Recommendation System is a machine learning application that predicts the most suitable crop for cultivation based on various environmental factors such as soil nutrients, temperature, and rainfall. This tool can assist farmers and agricultural researchers in making informed crop choices, thus helping optimize agricultural yield and sustainability.

Features

User-Friendly Web Interface: Built with Flask, HTML, and CSS to provide a simple yet effective platform for user interaction.
Input Validation: Ensures the environmental data inputs are within reasonable limits for better prediction accuracy.
Automated Crop Prediction: Based on soil and weather inputs, the system suggests a crop that is most suitable for the given conditions.
Image Display for Crop: Displays an image of the predicted crop, making the results visually intuitive.
Error Handling and Feedback: Provides feedback on any invalid input through error messages to guide users.

Technologies Used

Python: For backend development and ML model training
Flask: To serve the web interface
Machine Learning: Model built using algorithms such as Decision Trees for crop recommendation
HTML, CSS: Frontend for a responsive user interface
Pickle: To load the pre-trained mode

Model Training

The crop recommendation model was trained using a Decision Tree algorithm on a dataset containing environmental conditions (N, P, K, temperature, humidity, pH, rainfall) and corresponding crop labels. The trained model was serialized using Pickle and can be retrained with updated data as needed.
