from quart import Quart, jsonify, render_template, request
from quart_cors import cors 
import aiomysql
import torch
from src import get_contour, generator, option_to_meaning, value_to_point, connect, contrast
import os
from dotenv import load_dotenv

app = Quart(__name__)
app = cors(app)

load_dotenv()

async def get_db_connection():
    # Fetch values from environment variables
    db_host = os.getenv('DB_HOST', 'localhost')
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', 'idk_try_to_guess?')
    db_name = os.getenv('DB_NAME', 'db_name')
    db_min_size = int(os.getenv('DB_MIN_SIZE', 5))  
    db_max_size = int(os.getenv('DB_MAX_SIZE', 10)) 

    # Create the connection pool using .env
    return await aiomysql.create_pool(
        host=db_host,
        user=db_user,
        password=db_password,
        db=db_name,
        minsize=db_min_size,
        maxsize=db_max_size
    )

@app.route('/')
async def home():
    return await render_template("index.html")

@app.route('/from-data-to-image', methods=['POST'])
async def from_data_to_image():
    received_data = await request.get_json()
    print("Received JSON data:", received_data)
    
    text = option_to_meaning.option_2_meaning(received_data)

    sig_name = received_data.get("name")
    
    if(received_data.get("boss") == "edge"):
        sig_style = 1
    else:
        sig_style = 0

    if(received_data.get("symbol") == "omega"):
        sig_symbol = 1
    elif(received_data.get("symbol") == "loop"):
        sig_symbol = 2
    else:
        sig_symbol = 0
    
    if(received_data.get("tilt") == "tilt_up"):
        sig_tilt = True
    else:
        sig_tilt = False
    
    if(received_data.get("point") == "checked_point"):
        sig_dot = True
    else:
        sig_dot = False

    if(received_data.get("line") == "checked_line"):
        sig_line = True
    else:
        sig_line = False

    sig_data = await connect.v_concat(sig_name, sig_style, sig_symbol, sig_tilt, sig_dot, sig_line)

    print("angle: ",sig_data.get("angle")," tall_ratio: ",sig_data.get("tall_ratio")," distance: ",sig_data.get("distance"), " head_broken: ",sig_data.get("head_broken"), " head_cross: ",sig_data.get("head_cross"))

    points = {
        "point1": value_to_point.value_2_point1(sig_data.get("angle")),
        "point2": value_to_point.value_2_point2_3(sig_data.get("tall_ratio")),
        "point3": value_to_point.value_2_point2_3(sig_data.get("distance")),
        "point4": value_to_point.value_2_point4(sig_data.get("head_broken")),
        "point5": value_to_point.value_2_point5(sig_data.get("head_cross"), sig_data.get("head_is"))
    }

    print(points)
    
    try:
        # Assuming image is generated asynchronously
        base64_image = contrast.increase_contrast(sig_data.get('image'), 1.7)
        base64_image = f"data:image/png;base64,{base64_image}"
        
        return jsonify({
            "message": "Success",
            "image": base64_image,
            "text": text,
            "points": points
        })
    except Exception as e:
        return jsonify({"error": "Error reading image", "details": str(e)}), 500

async def insert_inquiry_to_db(data):
    pool = await get_db_connection()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor: 

            query = """
            INSERT INTO inquiries (
                category_1, category_2, category_3, category_4, image, 
                name_input, boss, tilt, symbol, checkbox_point, checkbox_line, 
                sig_point1, sig_point2, sig_point3, sig_point4, sig_point5, text
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert data to appropriate types
            params = (
                int(data.get("category_1", 0)), 
                int(data.get("category_2", 0)),
                int(data.get("category_3", 0)),
                int(data.get("category_4", 0)),
                data.get("image", ""),  # TEXT
                data.get("name_input", ""),  # VARCHAR(255)
                data.get("boss", ""),  # VARCHAR(255)
                data.get("tilt", ""),  # VARCHAR(255)
                data.get("symbol", ""),  # VARCHAR(255)
                data.get("checkbox_point", ""),  
                data.get("checkbox_line", ""),  
                int(data.get("sig_point1", 0)),  
                int(data.get("sig_point2", 0)),
                int(data.get("sig_point3", 0)),
                int(data.get("sig_point4", 0)),
                int(data.get("sig_point5", 0)),
                data.get("text", "")  # TEXT
            )
            print("Executing Query: ", query)
            print("Query Parameters: ", params)

            # Execute query
            await cursor.execute(query, params)
            await conn.commit()
            return cursor.lastrowid 

@app.route('/inquiry', methods=['POST'])
async def get_inquiry():
    received_data = await request.get_json()
    print("Received JSON data:", received_data)

    try:
        # Insert the inquiry data into the database
        inquiry_id = await insert_inquiry_to_db(received_data)

        return jsonify({
            "message": "Success",
            "receivedData": received_data,
            "inquiryId": inquiry_id,
        })
    except Exception as e:
        return jsonify({
            "error": "Error saving inquiry to the database",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, port=80)
    except Exception as e:
        print(f"App crashed: {e}")