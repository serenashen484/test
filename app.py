import streamlit as st
import time
import datetime
from PIL import Image
from google.cloud import firestore


class MyApp():
    ############################
    # private helper functions #
    ############################
    def __update_page(self, str):
        # -----------------------
        # update the page state
        # -----------------------
        st.session_state['page'] = str
    
    def __add_user(self, first_name, last_name, email):
        # -----------------------
        # add a new user to the database
        # -----------------------
        # calculate new user's userID (= last userID + 1)
        user_id_list = users["UserID"]
        users["UserID"].append(user_id_list[len(user_id_list) - 1] + 1)

        # add data given to database
        users["FirstName"].append(first_name)
        users["LastName"].append(last_name)
        users["Email"].append(email)

        ref.update(users)
    
    def __logout(self):
        # -----------------------
        # create a logout widget
        # -----------------------
        if st.button('Logout'):
            # clear all session states
            st.session_state['user_id'] = ''
            st.session_state['name'] = ''
            st.session_state['info'] = None

            # return to login page
            self.__update_page('login')
            st.experimental_rerun()

    # def __delete_account(self, user_id):
    #     # -----------------------
    #     # create a delete account expander
    #     # -----------------------
    #     with st.expander(f'**{"Delete My Account"}**'):
    #         # warning message, confirm checkbox, and delete button
    #         st.error(f'This action \***{"cannot"}**\* be undo. Please check the box below to delete.')
    #         cb = st.checkbox('Yes, I confirm')
    #         b = st.button(f'**{"Delete"}**')

    #         if cb and b:
    #             with st.empty():
    #                 with st.spinner("Deleting..."):
    #                     # delete from addresss
    #                     address_id = pd.read_sql('SELECT AddressID FROM users WHERE Username = ?',
    #                                             conn, params=(st.session_state['username'],))["AddressID"].item()
    #                     cur.execute('DELETE FROM address WHERE AddressID = ?', (address_id,))
    #                     # delete from food activity
    #                     cur.execute('DELETE FROM food_activity WHERE UserID = ?', (user_id,))
    #                     # delete from exercise activity
    #                     cur.execute('DELETE FROM exercise_activity WHERE UserID = ?', (user_id,))
    #                     # delete from users
    #                     cur.execute('DELETE FROM users WHERE UserID = ?', (user_id,))
    #                     conn.commit()
    #                     time.sleep(1)
    #                 with st.spinner("Redirecting to Login page..."):
    #                     # reset session_state and return to login page
    #                     st.session_state['username'] = ''
    #                     st.session_state['name'] = ''
    #                     st.session_state['info'] = None
    #                     self.__update_page('login')
    #                     time.sleep(2)
    #                     st.experimental_rerun()

    def __user_content(self):
        # -----------------------
        # main contents for users
        # -----------------------
        user_id = st.session_state['user_id']

        tabs = st.tabs([f'**{"Home"}**', f'**{"Book Lessons"}**', f'**{"Forms"}**', f'**{"My Account"}**'])

        # home tab
        with tabs[0]:
            st.header(f'Welcome back, *{st.session_state["name"]}*')
            
        # book lessons tab
        with tabs[1]:
            image = Image.open('schedule.jpg')
            st.image(image, caption='Summer 2023 shedule')

            # get lessons data
            lessons = ref.document("lessons").get().to_dict()
            dates = []
            
            for i in range(len(lessons["dates"])):
                if lessons["num"][i] > 0:
                    dates.append(lessons["dates"][i]  + " (" + str(lessons["num"][i]) + " available)")

            # booking form
            form = st.form("booking")
            form.subheader("Book Lessons")

            selected = form.multiselect('Dates', dates, key="selected")
            n = form.number_input('Number of students', key="student_num", min_value=1, step=1)
            names = form.text_input('Student name(s)', key="student_names", placeholder="e.g. Serena")
            c = form.checkbox('I have read, understand, and agree to the Terms and Conditions.', key="checkbox")

                # submit button
            if form.form_submit_button("Book"):
                enough_spots = True
                for d in selected:
                    if lessons["num"][lessons["dates"].index(d)] < n:
                        enough_spots = False

                # check conditions
                if len(selected) < 10:
                    st.error("Please select at least ten (10) lessons.")
                elif not names:
                    st.error("Please enter student name(s).")
                elif not c:
                    st.error("Please agree to the Terms and Conditions.")
                elif not enough_spots:
                    st.error("Lesson on " + d + " does not have enough spots left.")
                else:
                    st.success("Your lessons are booked!")
                    with st.spinner("Saving..."):
                        # insert data into database
                        doc = ref.document(st.session_state["user_id"])
                        if doc.get().exists:
                            users_file = doc.to_dict()
                        else:
                            doc.set({datetime.date.now().strftime("%Y-%m-%d, %H:%M:%S"): selected})

                        # update spots available
                        for d in selected:
                            lessons["num"][lessons["dates"].index(d)] -= n
                        lessons.update({"num": True})
                        time.sleep(2)
                    st.session_state['lessons_booked'] = True
                    st.experimental_rerun()

        # add exercise activity tab
        with tabs[2]:
            st.write('Under construction')
        #     # get exercise name list
        #     options = pd.read_sql('SELECT * FROM exercise',
        #                           conn)["NameOfExercise"].sort_values().to_list()
        #     options.insert(0, "Please type or select below")

        #     # fields for user to fill in
        #     ex_id = options.index(st.selectbox('Exercise Name', options, key="ex_name"))
        #     ex_date = st.date_input('Date', key="ex_date", max_value=datetime.datetime.today())
        #     ex_time = st.time_input('Time', key="ex_time")
        #     duration = st.number_input('Duration (minute)', min_value=0.0, max_value=1440.0, step=1.0, key="duration")
            
        #     # if save button is pressed
        #     if st.button('Save', key="ex_save"):
        #         # check if all fields are filled
        #         if not ex_id:
        #             st.error("Please select a valid exercise name.")
        #         elif duration == 0.0:
        #             st.error("Please enter a valid duration.")
        #         else:
        #             st.success("Saved!")
        #             with st.spinner("Refreshing in a second..."):
        #                 # generate new food activity id (= last id + 1)
        #                 ea_id_list = pd.read_sql('SELECT ExerciseActivityID FROM exercise_activity',
        #                                          conn)["ExerciseActivityID"].to_list()
        #                 if len(ea_id_list):
        #                     ea_id = ea_id_list[len(ea_id_list) - 1] + 1
        #                 else:
        #                     ea_id = 1

        #                 # convert date and time to string
        #                 dt = ex_date.strftime("%Y-%m-%d") + " " + ex_time.strftime("%H:%M")
                        
        #                 # insert data into database
        #                 cur.execute('INSERT INTO exercise_activity VALUES (?,?,?,?,?)',
        #                             (ea_id, user_id, ex_id, duration, dt))
        #                 conn.commit()
        #                 time.sleep(2)
        #             st.session_state['ex_saved'] = True
        #             st.experimental_rerun()

        # my account tab
        with tabs[3]:
            st.subheader("Personal Information")
            message = st.empty()
            
            # user_df = pd.read_sql('SELECT * FROM users WHERE UserID = ?', conn, params=(user_id,))

            # # if all data is saved in the database already, fetch data from the database
            # if user_df['Gender'].item():
            #     gender_index = gender_list.index(user_df['Gender'].item())
            #     birth = datetime.datetime.strptime(user_df['DateOfBirth'].item(), '%Y-%m-%d')
            #     phone = user_df['Phone'].item()
            #     weight = float(user_df['Weight'].item())
            #     height = float(user_df['Height'].item())

            #     address_id = user_df['AddressID'].item()
            #     address_df = pd.read_sql('SELECT * FROM address WHERE AddressID = ?', conn, params=(address_id,))
            #     address = address_df['Address'].item()
            #     zipcode = address_df['Zipcode'].item()
            #     country = address_df['Country'].item()
            # else:
            #     # warning message
            #     message.warning('Please update your personal information and complete your profile')

            #     # otherwise, set to default (empty strings or 0 for numbers)
            #     gender_index = 0
            #     birth = datetime.datetime.today()
            #     phone = ''
            #     weight = 0.0
            #     height = 0.0
            #     address_id = 0
            #     address = ''
            #     zipcode = ''
            #     country = ''

            # # view mode
            # if st.session_state['info'] == None or st.session_state['info'] == 'saved':
            #     # if the edit button is pressed, change the state and rerun
            #     if st.button('Edit'):
            #         st.session_state['info'] = 'edit'
            #         st.experimental_rerun()

            #     # all fields with the saved data fetched, not editable
            #     st.text_input('Username', st.session_state['username'], disabled=True)
            #     st.text_input('First Name', st.session_state["name"], disabled=True)
            #     st.text_input('Last Name', user_df['LastName'].item(), disabled=True)
            #     st.selectbox('Gender', gender_list, index=gender_index, disabled=True)
            #     st.date_input('Date of Birth', value=birth, disabled=True)
            #     st.text_input('Phone Number', phone, disabled=True)
            #     st.number_input('Weight (kg)', weight, step=1.0, disabled=True)
            #     st.number_input('Height (cm)', height, step=1.0, disabled=True)
            #     st.text_input('Home Address', address, disabled=True)
            #     st.text_input('Zip Code', zipcode, disabled=True)
            #     st.text_input('Country', country, disabled=True)
            # # edit mode
            # else:
            #     # remove warning message
            #     message.empty()

            #     # use a container so that error messages can be added below 'Save' button later
            #     c = st.container()
            #     save = c.button('Save')

            #     # all fields with the saved data fetched, record the new input
            #     st.text_input('Username', st.session_state['username'], disabled=True)
            #     first_name = st.text_input('First Name', st.session_state["name"], max_chars=50)
            #     last_name = st.text_input('Last Name', user_df['LastName'].item(), max_chars=50)
            #     gender = st.selectbox('Gender', gender_list, index=gender_index)
            #     birth = st.date_input('Date of Birth', value=birth, min_value=datetime.date(1900, 1, 1), max_value=datetime.datetime.today())
            #     phone = st.text_input('Phone Number', phone, max_chars=50)
            #     weight = st.number_input('Weight (kg)', value=weight, min_value=0.0, max_value=550.0, step=1.0)
            #     height = st.number_input('Height (cm)', value=height, min_value=0.0, max_value=300.0, step=1.0)
            #     address = st.text_input('Home Address', address, max_chars=50)                
            #     zipcode = st.text_input('Zip Code', zipcode, max_chars=50)
            #     country = st.text_input('Country', country, max_chars=50)
                
            #     # if save button is pressed
            #     if save:
            #         # check if all fields are filled
            #         if not first_name or not last_name:
            #             c.error("Please enter your full name")
            #         elif gender == 'Please type or select below':
            #             c.error("Please select your gender")
            #         elif not phone:
            #             c.error("Please enter your phone number")
            #         elif not weight:
            #             c.error("Please enter a valid weight")
            #         elif not height:
            #             c.error("Please enter a valid height")
            #         elif not address:
            #             c.error("Please enter your home address")
            #         elif not zipcode:
            #             c.error("Please enter your zip code")
            #         elif not country:
            #             c.error("Please enter your country of residence")
            #         else:
            #             c.success("Saved!")
            #             with c:
            #                 with st.spinner("Refreshing in a second..."):
            #                     # if there is a valid address id, update the address saved
            #                     if address_id:
            #                         cur.execute('UPDATE address SET Address = ?, Zipcode = ?, Country = ? WHERE AddressID = ?',
            #                                     (address, zipcode, country, address_id))
            #                     else:
            #                         # generate new address id (= last id + 1)
            #                         address_id_list = pd.read_sql('SELECT AddressID FROM address', conn)["AddressID"].to_list()
            #                         address_id = address_id_list[len(address_id_list) - 1] + 1
            #                         cur.execute('INSERT INTO address VALUES (?,?,?,?)', (address_id, address, zipcode, country))
                                
            #                     # update all data in the database
            #                     cur.execute('UPDATE users SET FirstName = ?, LastName = ?, Gender = ?, DateOfBirth = ?, '
            #                                 'Phone = ?, Weight = ?, Height = ?, AddressID = ? WHERE UserID = ?',
            #                                 (first_name, last_name, gender, birth, phone, weight, height, address_id, user_id))
            #                     conn.commit()

            #                     # change state to saved and rerun
            #                     st.session_state['info'] = 'saved'
            #                     time.sleep(2)
            #                     st.experimental_rerun()
            
            # reset password and logout button
            st.write('')
            st.subheader('')
            self.__logout()

            # # delete my account expander
            # st.write('')
            # st.subheader('')
            # self.__delete_account(user_id)

    ####################
    # public functions #
    ####################
    def login(self):
        # -----------------------
        # create a login page
        # -----------------------
        # form and all fields
        form = st.form('login')
        form.subheader('Login')
        email = form.text_input('Email', placeholder="e.g. swim.bluefin@gmail.com").lower()

        # if login button is pressed
        if form.form_submit_button('Login'):
            # check if all fields are filled
            if not email:
                form.warning('Please enter your email')
            # check credentials (helpder function)
            elif email not in users["Email"]:
                form.error('Incorrect email')
            else:
                # update session state with current user's info
                st.session_state['user_id'] = users["UserID"][users["Email"].index(email)]
                st.session_state['name'] = users["FirstName"][users["Email"].index(email)]

                # set page to home and rerun
                self.__update_page('home')
                st.experimental_rerun()
        
        # button leads to sign up page
        st.button("Sign Up", on_click=self.__update_page, args=('signup',))

    def signup(self):
        # -----------------------
        # create a signup page
        # -----------------------
        # form and all fields
        form = st.form('signup')
        form.subheader('Create a new account')
        first_name = form.text_input('First Name', placeholder="e.g. Serena", max_chars=50)
        last_name = form.text_input('Last Name', placeholder="e.g. Shen", max_chars=50)
        email = form.text_input('Email', placeholder="e.g. swim.bluefin@gmail.com").lower()
        
        # if sign up button is pressed
        if form.form_submit_button('Sign Up'):
            # check requirements:
            #   - all fields are filled
            #   - email not already exits
            if not first_name or not last_name:
                form.warning('Please enter your full name')
            elif not email:
                form.warning('Please enter a email')
            elif email in users["Email"]:
                form.error('Email already taken, please use another email')
            else:
                # add new user to database (helpder function)
                self.__add_user(first_name, last_name, email)
                form.success('User registered successfully')

        # back to login button
        st.button("Back to Login", on_click=self.__update_page, args=('login',))
    
    def home(self):
        # -----------------------
        # create a home page
        # -----------------------
        # welcome message
        st.title(f'Bluefin & Jessica Swim School')
        st.caption(datetime.date.today().strftime("%B %d, %Y"))
        st.write('')

        # content
        # if st.session_state['user_id'] == 0:
        #     self.__admin_content()
        # else:
        self.__user_content()
    
    def booking_clear(self):
        # -----------------------
        # clear all fields under the 'Add Food Activity' tab
        # -----------------------
        st.session_state['selected'] = []
        st.session_state['student_num'] = 1
        st.session_state['student_names'] = ''
        st.session_state['checkbox'] = False
        st.session_state['lessons_booked'] = False



# main function
if __name__ == "__main__":
    # make pages wide
    st.set_page_config(page_title="Bluefin & Jessica Swim School", page_icon="🏊🏻", layout="wide")
    myapp = MyApp()

    # Authenticate to Firestore with the JSON account key.
    db = firestore.Client.from_service_account_json("firebase-key.json")

    # Create a reference to the Google post.
    ref = db.collection("data")
    users = ref.document("users").get().to_dict()
    
    # initialize at the beginning of the program
    if not st.session_state:
        # initialize session state
        st.session_state['page'] = 'login'
        st.session_state['lessons_booked'] = False
        st.session_state['ex_saved'] = False
        st.session_state['info'] = None

    # clear fields under some tabs if needed
    if st.session_state['lessons_booked']:
        myapp.booking_clear()
    elif st.session_state['ex_saved']:
        myapp.ex_clear()

    # pages
    if st.session_state['page'] == 'login':
        myapp.login()
    elif st.session_state['page'] == 'signup':
        myapp.signup()
    elif st.session_state['page'] == 'home':
        myapp.home()