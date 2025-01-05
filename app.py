#Import libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import re
from numpy import polyfit, poly1d



# Load the dataset from spam_emails.csv
@st.cache_data
def load_data():
    data = pd.read_csv('spam_emails.csv')  
    return data

data = load_data()





# Train the model using TF-IDF and Random Forest Classifier
@st.cache_data
def train_my_detector(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['text']).toarray()
    y = data['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, vectorizer

model, vectorizer = train_my_detector(data)





# The preloaded examples that the user can select to test
example_emails = {
    "Select an Example": "",
   
   
   
    "Spam Email Example 1": """Subject: You’ve Won Big!
    
Dear Customer,

Congratulations! You are the lucky winner of our $1,000,000 giveaway! This is your chance to make your dreams come true. All you need to do is click the link below and complete a quick verification process to claim your prize. But hurry! This offer expires in 24 hours.

[Claim Your Prize Now]

Don’t miss out on this life-changing opportunity!

Best Regards,  
The Lucky Oyler Draw Team""",
    






    "Spam Email Example 2": """Subject: Urgent: Verify Your Account Now!

Dear User,

We detected unusual activity in your bank account and have temporarily suspended access to protect your funds. To restore access, verify your identity immediately by clicking the secure link below:

[Verify My Account]

Failure to act within 48 hours will result in permanent account deactivation. Thank you for your prompt attention to this matter.

Sincerely,  
Trevor Security Team""",
   
   
   



   
    "Not Spam Email Example 1": """Subject: Team Meeting Reminder

Hi Team,

This is a reminder about tomorrow's meeting at 10:00 AM in the main conference room. We will discuss project milestones and assign roles for the next quarter.

Please let me know if you'd like to add any agenda items.

Best regards,  
Trevor Oyler  
Project Manager""",
   
   
   
   


   
    "Not Spam Email Example 2": """Subject: Your Order Has Been Shipped

Dear Customer,

Your order (#965472) has been shipped and is on its way to you. Track your package using the tracking number: 11574821.

Your estimated delivery date is January 10th. Contact our support team if you have any questions.

Thank you for shopping with us!

Best regards,  
Customer Support Team  
ShopWithTrevor Inc."""
}




#Website Title
st.title("Spam Email Detector")




# introduction text
st.markdown(
    """
    <p style="font-size:18px;">
    Welcome to the Spam Email Detector! This tool uses Machine Learning to classify emails as either 
    spam or not spam. Use the navigation options below to explore more.
    </p>
    """,
    unsafe_allow_html=True,
)




# navigation text made bold
st.markdown(
    """
    <p style="font-size:18px; font-weight:bold;">Navigate to:</p>
    """,
    unsafe_allow_html=True,
)




#navigation buttons
page = st.radio(
    "Page Navigation",
    ["Spam Email Detector", "Data Visualizations"],
    horizontal=True,
    label_visibility="collapsed"
)





#Main page
if page == "Spam Email Detector":
    st.header("Spam Email Detector")
    st.write(
        """
        Enter an email in the box below, or select one of the preloaded examples to test the spam detector. The preloaded examples were not included in training.
        """
    )


    
    #Dropdown for preloaded emails
    selected_example = st.selectbox("Choose a preloaded email:", list(example_emails.keys()))
    



    #text box for user input
    user_input = st.text_area(
        "Enter email here:",
        value=example_emails[selected_example],
        height=300
    )



    #button to classify emails
    if st.button("Classify Email"):
        if user_input.strip():
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            


            # Displays the prediction with spam as red and no spam detected as green
            if prediction == 1:
                st.markdown(
                    '<p style="color:red; font-size:20px;">Prediction: <strong>Spam Detected</strong></p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<p style="color:green; font-size:20px;">Prediction: <strong>No Spam Detected</strong></p>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter text.")



# Data Visualization Page
elif page == "Data Visualizations":
    st.header("Visualizations for Spam Detection")
    st.write("Explore visual representations from the dataset used to train this Machine Learning model.")



    #Pie Chart to show spam vs non spam emails
    st.subheader("Spam vs. Not Spam Distribution")
    st.write("This pie chart shows the proportion of spam and non-spam emails in the dataset.")
    spam_counts = data['spam'].value_counts()
    spam_counts.rename({0: 'Not Spam', 1: 'Spam'}).plot.pie(autopct='%1.1f%%', startangle=90, figsize=(5, 5))
    plt.title("Spam vs. Not Spam")
    plt.ylabel("")
    st.pyplot(plt)



    # Bar plot graph for Text Length in spam vs non spam emails
    st.subheader("Text Length in Spam vs. Not Spam Emails")
    st.write("This bar plot graph shows the average text length in spam vs. not spam emails.")
    data['text_length'] = data['text'].apply(len)
    avg_text_length = data.groupby('spam')['text_length'].mean()
    avg_text_length.index = ['Not Spam', 'Spam']
    st.bar_chart(avg_text_length)



    # Scatterplot to show words vs special characterrs
    st.subheader("Words vs. Special Characters")
    st.write(
        """
        This scatterplot compares the number of words vs. special characters in spam and non spam emails.
        """
    )
    data['word_count'] = data['text'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
    data['special_char_count'] = data['text'].apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))

    plt.figure(figsize=(10, 6))
    for spam_label, label_name, color in [(0, 'Not Spam', 'blue'), (1, 'Spam', 'orange')]:
        subset = data[data['spam'] == spam_label]
        plt.scatter(subset['word_count'], subset['special_char_count'], label=label_name, alpha=0.6, color=color, s=50)
    x = data['word_count']
    y = data['special_char_count']
    trend = polyfit(x, y, 1)
    trendline = poly1d(trend)
    plt.plot(x, trendline(x), color='red', linestyle='--', label='Trendline')
    plt.title("Word Count vs. Special Characters")
    plt.xlabel("Word Count")
    plt.ylabel("Special Character Count")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
