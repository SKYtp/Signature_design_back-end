def option_2_meaning(data):
    text = ""
    text += "• เริ่มเขียนจากพยัญชนะนำตำแหน่งที่ 1: ให้เขียนพยัญชนะตัวแรกของชื่อให้มีขนาดใหญ่และชัดเจน เพื่อสะท้อนถึงความคิดสร้างสรรค์และความมั่นใจ ควรหลีกเลี่ยงการเขียนตัวอักษรที่ไม่สมบูรณ์หรือมีการเติมหรือลดลงที่ไม่จำเป็น\n\n"
    text += "• เว้นวรรคระหว่างตำแหน่งประธานและบริวาร: ตัวอักษรแรกควรมีขนาดใหญ่กว่าและไม่ติดกับตัวอักษรที่สองและให้อยู่ในระนาบเดียวกัน เพื่อไม่ให้บริวารมีอิทธิพลเหนือกว่าตัวเรา และการเว้นวรรคที่เหมาะสมสามารถช่วยป้องกันไม่ให้บริวารมีอิทธิพลเหนือตัวเราได้\n\n"
    text += "• การเว้นช่องไฟ: ขนาดช่องไฟระหว่างตัวอักษรตัวแรกกับตัวอักษรตัวที่สอง ควรมีขนาดเศษหนึ่งส่วนสองของตัวอักษร ซึ่งจะช่วยให้ชีวิตดำเนินไปอย่างราบรื่นและไม่มีปัญหาจากคนรอบข้าง\n\n"
    if(data.get("boss") == 'round'):
        text += "• ตัวอักษรในลายเซ็นที่มีรูปทรงโค้งมนและไม่มีมุมสื่อถึงความอ่อนโยน, เข้าถึงได้ง่าย, และความมีไหวพริบในการปรับตัว ตัวอักษรที่มีโค้งมนช่วยบรรเทาความรุนแรงและส่งเสริมให้เกิดความสัมพันธ์ที่ดีกับผู้อื่น และมักสื่อถึงบุคคลที่มีความอบอุ่น, ความใส่ใจ, และความเข้าใจต่อผู้อื่น รูปแบบนี้เหมาะกับบุคคลในสายอาชีพที่ต้องการการติดต่อสัมพันธ์และการดูแลเอาใจใส่ต่อผู้อื่น\n\n"
    elif(data.get("boss") == 'edge'):
        text += "• ตัวอักษรในลายเซ็นที่มีรูปทรงแหลมคมหรือเหลี่ยมมุมสื่อถึงลักษณะนิสัยที่เกี่ยวข้องกับความเด็ดขาด, ความกล้าหาญ, ความแข็งแกร่ง, ความมุ่งมั่น และมีการตัดสินใจที่รวดเร็ว รูปแบบนี้อาจพบได้บ่อยในบุคคลที่ทำงานในสายอาชีพที่ต้องใช้ความเด็ดขาดและการตัดสินใจที่รวดเร็ว\n\n"
    if(data.get("tilt") == 'none'):
        text += "• แนวระนาบ: ลายเซ็นนี้สะท้อนถึงความมั่นคงทางอารมณ์และจิตใจ เป็นคนที่มีวินัย และสามารถควบคุมอารมณ์และการกระทำของตัวเองได้ดี\n\n"
    elif(data.get("tilt") == 'tilt_up'):
        text += "• เอียงขึ้น: ลายเซ็นที่เอียงขึ้นบ่งบอกถึงความทะเยอทะยานและความกล้า คุณมีทัศนคติที่เชื่อมั่นในการดำเนินชีวิตไปข้างหน้าและไม่หวั่นเกรงต่ออุปสรรค มีความมั่นใจที่จะเผชิญหน้ากับทุกสถานการณ์\n\n"
    if(data.get("symbol") == 'omega'):
        text += " โอเมก้า: บ่งบอกถึงความเป็นนักเจรจาต่อรองที่ดี มีความสามารถในการโน้มน้าวใจผู้คน โดยโอเมก้าที่ดีควรเขียนในรูปแบบที่คล้ายกับตัวอักษร M หรือ W เพื่อแสดงถึงโอเมก้าคว่ำและโอเมก้าหงาย ในการเขียนโอเมก้าจะต้องไม่เขียนเป็นตัวโอเมก้าตรงๆ เพราะนั่นคือการจงใจเจตนาสร้าง ซึ่งจะต้องเป็นการเขียนโอเมก้าโดยสัญลักษณ์ของมันเอง•\n\n"
    elif(data.get("symbol") == 'loop'):
        text += "• ห่วง: มีความเอื้ออาทรต่อคนใกล้ชิด โดยห่วงที่ดีจะมีลักษณะเป็นตัว L ที่ลากแล้วเป็นห่วง ไม่ควรขีดเป็นเหมือน T ซึ่งจะกลายเป็นห่วงเทียมบ่งบอกถึงความเสแสร้ง ไม่จริงใจ\n\n"
    if(data.get("point") == 'checked_point'):
        text += "• จุด: บ่งบอกถึงความมั่นใจและความเด็ดขาดในการตัดสินใจ โดยลักษณะของจุดที่ดีจะต้องนำมาใช้เป็นจุดที่ต่อจากชื่อเมื่อเราเซ็นชื่อเสร็จและต้องเป็นจุดที่สร้างขึ้นมาใหม่ไม่ใช่แทนตัวอักษรหรือสระ\n\n"
    if(data.get("line") == 'checked_line'):
        text += "• เส้น: บ่งบอกถึงการมีคนสนับสนุน เกื้อหนุน โดยเส้นที่ดีควรจะมีตอนท้ายชื่อและยาวพอประมาณและควรจะมีเส้นเอียงขึ้น 45 องศา\n\n"
    if(data.get("circle") == 'checked_circle'):
        text += "• วงกลม: บ่งบอกถึงความคิดสร้างสรรค์และมุมมองที่กว้างขวาง คุณเป็นคนที่มีแนวคิดเปิดกว้างและสามารถคิดค้นสิ่งใหม่ๆ ได้ตลอดเวลา โดยวงกลมที่ดีไม่ควรมีขนาดที่ใหญ่ เนื่องจากจะสื่อถึงภาระที่ถูกรับมอบหมายจากคนที่เหนือกว่า เช่น ผู้บังคับบัญชา\n\n"
    return text