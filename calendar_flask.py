from flask import Flask, jsonify, request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import datetime
import pickle
import traceback

app = Flask(__name__)

# Path to your client secret JSON file from Google Cloud Console
CLIENT_SECRET_FILE = 'client_secret.json'
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def get_calendar_service():
    creds = None
    # Check if token.pickle file exists (contains access and refresh tokens)
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If no credentials available, login with OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=9090)  # Specify a fixed port
        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('calendar', 'v3', credentials=creds)

@app.route('/events', methods=['GET'])
def get_events():
    try:
        service = get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        end_of_day = (datetime.datetime.utcnow() + datetime.timedelta(hours=23, minutes=59)).isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary', timeMin=now, timeMax=end_of_day,
            singleEvents=True, orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        formatted_events = [
            {
                'summary': event.get('summary', 'No Title'),
                'start': event['start'].get('dateTime', event['start'].get('date')).split('T')[1][:5],
                'end': event['end'].get('dateTime', event['end'].get('date')).split('T')[1][:5]
            }
            for event in events
        ]

        return jsonify({'events': formatted_events})

    except Exception as e:
        # Log the detailed traceback to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
