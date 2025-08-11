import pandas as pd
import numpy as np
import re
import joblib
from email.parser import Parser
from email.policy import default

from bs4 import BeautifulSoup
from urllib.parse import urlparse

def load_models():
    try:
        model = joblib.load('bestmodel3_random_forest.pkl')
        
        scaler = joblib.load('scaler3.pkl')
        
        print("Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def extract_basic_features(email_content):
    if isinstance(email_content, str):
        msg = Parser(policy=default).parsestr(email_content)
    else:
        msg = email_content
    
    from_address = msg.get('From', '')
    to_address = msg.get('To', '')
    subject = msg.get('Subject', '')
    reply_to = msg.get('Reply-To', '')
    date = msg.get('Date', '')
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" or content_type == "text/html":
                try:
                    body += part.get_payload(decode=True).decode()
                except:
                    body += str(part.get_payload())
    else:
        body = msg.get_payload(decode=True).decode() if msg.get_payload() else ""
    
    if "<html" in body.lower() or "<body" in body.lower():
        try:
            soup = BeautifulSoup(body, 'html.parser')
            body_text = soup.get_text()
        except:
            body_text = body
    else:
        body_text = body
    
    return {
        'from_address': from_address,
        'to_address': to_address,
        'subject': subject,
        'reply_to': reply_to,
        'date': date,
        'body': body,
        'body_text': body_text
    }

def analyze_urls(text):
    
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]*\.[^\s<>"]+\b'
    urls = re.findall(url_pattern, text)
    
    url_features = {
        'number_of_urls': len(urls),
        'max_url_length': max([len(url) for url in urls]) if urls else 0,
        'contains_ip_urls': 0,
        'suspicious_tld': 0
    }
    
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.xyz', '.pw']
    
    for url in urls:
        if re.search(r'https?://\d+\.\d+\.\d+\.\d+', url):
            url_features['contains_ip_urls'] = 1
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or parsed_url.path
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            url_features['suspicious_tld'] = 1
    
    return url_features

def analyze_sender(from_address, reply_to):
    """
    Analyze sender information
    """
    sender_features = {
        'from_domain_match': 1,
        'reply_to_mismatch': 0   
    }
    
    from_domain = ''
    reply_domain = ''
    
    try:
        from_match = re.search(r'@([\w.-]+)', from_address)
        from_domain = from_match.group(1) if from_match else ''
        
        if reply_to:
            reply_match = re.search(r'@([\w.-]+)', reply_to)
            reply_domain = reply_match.group(1) if reply_match else ''
            
            if from_domain and reply_domain and from_domain != reply_domain:
                sender_features['reply_to_mismatch'] = 1
        
        if '<' in from_address and '@' in from_address:
            display_name = from_address.split('<')[0].strip().lower()
            if display_name:
                domain_words = from_domain.split('.')
                if any(domain_word in display_name for domain_word in domain_words):
                    if not any(domain_word in from_domain for domain_word in display_name.split()):
                        sender_features['from_domain_match'] = 0
    except:
        pass
    
    return sender_features

def analyze_content(subject, body_text):
    """
    Analyze email content for suspicious elements
    """
    content_features = {
        'subject_length': len(subject),
        'subject_word_length': len(subject.split()),
        'body_length': len(body_text),
        'body_word_length': len(body_text.split()),
        'has_urgency_terms': 0,
        'has_sensitive_terms': 0,
        'suspicious_subject': 0,
        'number_of_misspelled_words': 0  
    }
    
    urgency_terms = ['urgent', 'immediate', 'alert', 'verify', 'suspended', 'action required', 
                     'attention', 'important', 'update required', 'security alert']
    if any(term in subject.lower() or term in body_text.lower() for term in urgency_terms):
        content_features['has_urgency_terms'] = 1
    
    sensitive_terms = ['password', 'credit card', 'ssn', 'social security', 'bank account', 
                       'verify your account', 'confirm your details', 'login', 'sign in']
    if any(term in subject.lower() or term in body_text.lower() for term in sensitive_terms):
        content_features['has_sensitive_terms'] = 1
    
    suspicious_subject_terms = ['your account', 'verify', 'update', 'suspended', 'unusual activity', 
                                'security', 'login', 'password']
    if any(term in subject.lower() for term in suspicious_subject_terms):
        content_features['suspicious_subject'] = 1
    
    common_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'you', 'that', 'it', 
                       'for', 'with', 'on', 'as', 'are', 'this', 'your', 'have', 'be', 'or'])
    body_words = re.findall(r'\b[a-zA-Z]{2,}\b', body_text.lower())
    suspicious_words = [w for w in body_words if len(w) > 3 and w not in common_words 
                        and not re.match(r'^[a-z]+$', w) and re.search(r'[a-z][A-Z]|[A-Z]{2,}', w)]
    content_features['number_of_misspelled_words'] = len(suspicious_words)
    
    return content_features

def analyze_attachments(msg):
    """
    Analyze email attachments
    """
    attachment_features = {
        'has_attachment': 0,
        'suspicious_attachment_type': 0,
        'number_of_attachments': 0
    }
    
    suspicious_extensions = ['.exe', '.zip', '.jar', '.js', '.bat', '.com', '.cmd', '.scr', '.vbs', '.ps1']
    
    if msg.is_multipart():
        for part in msg.walk():
            filename = part.get_filename()
            if filename:
                attachment_features['has_attachment'] = 1
                attachment_features['number_of_attachments'] += 1
                
                if any(filename.lower().endswith(ext) for ext in suspicious_extensions):
                    attachment_features['suspicious_attachment_type'] = 1
    
    return attachment_features

def analyze_html_content(body):
    """
    Analyze HTML content for phishing indicators
    """
    html_features = {
        'number_of_images': 0,
        'number_of_external_resources': 0,
        'contains_base64_content': 0
    }
    
    if "<html" in body.lower() or "<body" in body.lower():
        try:
            soup = BeautifulSoup(body, 'html.parser')
            
            html_features['number_of_images'] = len(soup.find_all('img'))
            
            external_resources = soup.find_all(['link', 'script', 'img', 'iframe'])
            external_count = 0
            for resource in external_resources:
                if resource.get('src') and ('http' in resource.get('src') or '//' in resource.get('src')):
                    external_count += 1
                elif resource.get('href') and ('http' in resource.get('href') or '//' in resource.get('href')):
                    external_count += 1
            
            html_features['number_of_external_resources'] = external_count
            
            if 'base64' in body:
                html_features['contains_base64_content'] = 1
        except:
            pass
    
    return html_features

def extract_email_features_from_input():
    """
    Collect email information from user input and extract features
    """
    print("\n===== Email Phishing Detection System =====\n")
    print("Please enter the email information:\n")
    
    from_address = input("From address (e.g., 'John Doe <john@example.com>'): ")
    to_address = input("To address: ")
    reply_to = input("Reply-To address (leave blank if none): ")
    subject = input("Subject: ")
    
    print("\nEnter email body text (press Enter twice when finished):")
    body_lines = []
    while True:
        line = input()
        if line == "":
            break
        body_lines.append(line)
    body = "\n".join(body_lines)
    
    email_text = f"From: {from_address}\nTo: {to_address}\n"
    if reply_to:
        email_text += f"Reply-To: {reply_to}\n"
    email_text += f"Subject: {subject}\n\n{body}"
    
    msg = Parser(policy=default).parsestr(email_text)
    
    basic_features = extract_basic_features(msg)
    
    url_features = analyze_urls(body)
    
    sender_features = analyze_sender(from_address, reply_to)
    
    content_features = analyze_content(subject, basic_features['body_text'])
    
    attachment_features = {
        'has_attachment': int(input("\nDoes the email have attachments? (1 for yes, 0 for no): ")),
        'suspicious_attachment_type': 0,
        'number_of_attachments': 0
    }
    
    if attachment_features['has_attachment'] == 1:
        attachment_features['number_of_attachments'] = int(input("Number of attachments: "))
        attachment_features['suspicious_attachment_type'] = int(input("Are there suspicious attachments (.exe, .zip, etc.)? (1 for yes, 0 for no): "))
    
    html_features = {
        'number_of_images': int(input("Number of images in the email: ")),
        'number_of_external_resources': int(input("Number of external resources (links to other websites): ")),
        'contains_base64_content': int(input("Contains encoded content (base64)? (1 for yes, 0 for no): "))
    }
    
    all_features = {**url_features, **sender_features, **content_features, 
                    **attachment_features, **html_features}
    
    all_features['subject_word_count'] = all_features['subject_length'] / (all_features['subject_word_length'] + 1)
    all_features['body_word_count'] = all_features['body_length'] / (all_features['body_word_length'] + 1)
    all_features['url_to_text_ratio'] = all_features['number_of_urls'] / (all_features['body_length'] + 1)
    all_features['image_to_text_ratio'] = all_features['number_of_images'] / (all_features['body_length'] + 1)
    all_features['long_urls_present'] = 1 if all_features['max_url_length'] > 100 else 0
    all_features['external_resources'] = 1 if all_features['number_of_external_resources'] > 0 else 0
    
    all_features['email_risk_score'] = (
        all_features['has_urgency_terms'] * 2.5 +
        all_features['has_sensitive_terms'] * 2.0 +
        all_features['suspicious_subject'] * 1.8 +
        all_features['suspicious_tld'] * 2.2 +
        (all_features['number_of_urls'] > 3) * 1.5 +
        all_features['has_attachment'] * 1.7 +
        all_features['suspicious_attachment_type'] * 2.8 +
        all_features['from_domain_match'] * (-2.0) +
        all_features['reply_to_mismatch'] * 2.3 +
        all_features['contains_ip_urls'] * 2.5 +
        all_features['contains_base64_content'] * 2.0 +
        all_features['long_urls_present'] * 1.3 +
        all_features['external_resources'] * 1.2 +
        (all_features['number_of_misspelled_words'] > 5) * 1.5
    )
    
    return all_features, basic_features

def apply_email_rules(prediction, features):
    """
    Apply heuristic rules to enhance model predictions for email phishing detection
    """
    if features['has_urgency_terms'] == 1 and features['has_sensitive_terms'] == 1:
        return 1, "Urgent language combined with requests for sensitive information"
    
    if features['from_domain_match'] == 0 and features['number_of_urls'] > 0:
        return 1, "Sender domain mismatch with suspicious URLs"
    
    if features['suspicious_attachment_type'] == 1:
        return 1, "Contains suspicious attachment types"
    
    if features['contains_ip_urls'] == 1:
        return 1, "Contains links with raw IP addresses"
    
    if features['reply_to_mismatch'] == 1:
        return 1, "Reply-to address differs from sender address"
    
    if features['email_risk_score'] > 8:
        return 1, "High overall phishing risk score"
    
    if features['number_of_urls'] > 2 and features['suspicious_tld'] == 1:
        return 1, "Multiple URLs with suspicious top-level domains"
    
    return prediction, "Based on machine learning model evaluation"

def main():
    model, scaler = load_models()
    if model is None or scaler is None:
        print("Error: Could not load model or scaler. Continuing with rule-based detection only.")
    
    features, basic_info = extract_email_features_from_input()
    
    features_df = pd.DataFrame([features])
    
    print("\n===== Email Analysis Summary =====")
    print(f"From: {basic_info['from_address']}")
    print(f"Subject: {basic_info['subject']}")
    print(f"URLs detected: {features['number_of_urls']}")
    print(f"Attachments: {features['number_of_attachments']}")
    print(f"Risk indicators detected:")
    
    risk_indicators = []
    if features['has_urgency_terms'] == 1:
        risk_indicators.append("- Urgent/time-sensitive language")
    if features['has_sensitive_terms'] == 1:
        risk_indicators.append("- Requests for sensitive information")
    if features['suspicious_tld'] == 1:
        risk_indicators.append("- Suspicious URL domains")
    if features['contains_ip_urls'] == 1:
        risk_indicators.append("- URLs with raw IP addresses")
    if features['reply_to_mismatch'] == 1:
        risk_indicators.append("- Reply-to address differs from sender")
    if features['from_domain_match'] == 0:
        risk_indicators.append("- Sender name/domain mismatch")
    if features['suspicious_attachment_type'] == 1:
        risk_indicators.append("- Suspicious attachment types")
    if features['contains_base64_content'] == 1:
        risk_indicators.append("- Contains encoded (base64) content")
    
    if risk_indicators:
        for indicator in risk_indicators:
            print(indicator)
    else:
        print("- No major risk indicators detected")
    
    prediction = 0
    confidence = 0
    model_used = "Rule-based only"
    
    if model is not None and scaler is not None:
        try:
            features_scaled = scaler.transform(features_df)
            
            prediction_proba = model.predict_proba(features_scaled)[0]
            raw_prediction = model.predict(features_scaled)[0]
            confidence = prediction_proba[1] if raw_prediction == 1 else prediction_proba[0]
            
            prediction, reason = apply_email_rules(raw_prediction, features)
            model_used = "Machine learning + Rules"
            
        except Exception as e:
            print(f"\nWarning: Error making prediction with model: {e}")
            print("Falling back to rule-based detection...")
            
            prediction, reason = apply_email_rules(0, features)
    else:
        prediction, reason = apply_email_rules(0, features)
    
    print("\n===== Detection Result =====")
    print(f"Detection method: {model_used}")
    
    if prediction == 1:
        print("\n PHISHING EMAIL DETECTED ")
        print(f"Reason: {reason}")
        print(f"Risk score: {features['email_risk_score']:.2f}/15.0")
        if model is not None and 'confidence' in locals():
            print(f"Confidence: {confidence * 100:.2f}%")
        
        print("\nRecommendation: Do not respond to this email, click any links, or open attachments.")
        print("Report this email to your IT security team or email provider.")
    else:
        print("\n✓ Email appears legitimate")
        print(f"Reason: {reason}")
        print(f"Risk score: {features['email_risk_score']:.2f}/15.0")
        if model is not None and 'confidence' in locals():
            print(f"Confidence: {confidence * 100:.2f}%")
        
        print("\nNote: Always exercise caution when handling emails, especially those requesting sensitive information.")

if __name__ == "__main__":
    main()