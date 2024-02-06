'''
This file contains the login function for the streamlit app.
'''
import yaml
from yaml import SafeLoader
from streamlit_authenticator import Authenticate
import streamlit as st

CONFIG_FILE = "config.yaml"
def login(render_func):
    '''
    This function is used to login to the streamlit app.
    '''
    with open(CONFIG_FILE, encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    c1, c2 = st.columns(2)
    with c1:
        name, authentication_status, _ = authenticator.login()
    if authentication_status:
        with c1:
            authenticator.logout('Logout', 'main')

        c2.write(f'Welcome *{name}*')
        render_func()

    else: # no login attempt
        c1.warning('Please enter your username and password')
