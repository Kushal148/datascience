import serial               
from time import sleep
import time
import webbrowser           
import sys
import smtplib, ssl

#-------------------------------------------------------------
def mail_send():
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "pmotagi0710@gmail.com"  # Enter your address
    receiver_email = "kushal14822@gmail.com"  # Enter receiver address
    password = "T2MT@RRBFZW3"
    message = """ \Subject: Blind Assist

    {}""".format(map_link)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
#--------------------------------------------------------------

def GPS_Info():
    global NMEA_buff
    global lat_in_degrees
    global long_in_degrees
    nmea_time = []
    nmea_latitude = []
    nmea_longitude = []
    nmea_time = NMEA_buff[0]                    #extract time from GPGGA string
    nmea_latitude = NMEA_buff[1]                #extract latitude from GPGGA string
    nmea_longitude = NMEA_buff[3]               #extract longitude from GPGGA string
    
    print("NMEA Time: ", nmea_time,'\n')
    print ("NMEA Latitude:", nmea_latitude,"NMEA Longitude:", nmea_longitude,'\n')
    
    lat = float(nmea_latitude)                  #convert string into float for calculation
    longi = float(nmea_longitude)               #convertr string into float for calculation
    
    lat_in_degrees = convert_to_degrees(lat)    #get latitude in degree decimal format
    long_in_degrees = convert_to_degrees(longi) #get longitude in degree decimal format
    
#convert raw NMEA string into degree decimal format   
def convert_to_degrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    mm_mmmm = (decimal_value - int(decimal_value))/0.6
    position = degrees + mm_mmmm
    position = "%.4f" %(position)
    return position
    


gpgga_info = "$GPGGA,"
ser = serial.Serial('/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0', baudrate=9600, timeout=1.0)
GPGGA_buffer = 0
NMEA_buff = 0
lat_in_degrees = 0
long_in_degrees = 0


while True:
        received_data = (str)(ser.readline())                   #read NMEA string received
        GPGGA_data_available = received_data.find(gpgga_info)   #check for NMEA GPGGA string                 
        if (GPGGA_data_available>0):
            GPGGA_buffer = received_data.split("$GPGGA,",1)[1]  #store data coming after "$GPGGA," string 
            NMEA_buff = (GPGGA_buffer.split(','))               #store comma separated data in buffer
            GPS_Info()                                          #get time, latitude, longitude
 
            x="lat in degrees:", lat_in_degrees," long in degree: ", long_in_degrees, '\n'
            print(x)

            map_link = 'http://maps.google.com/?q=' + lat_in_degrees + ',' + long_in_degrees    #create link to plot location on Google map
            mail_send()
            time.sleep(300) #in seconds, change as needed
            print("------------------------------------------------------------\n")
