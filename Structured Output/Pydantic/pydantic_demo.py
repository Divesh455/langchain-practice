from pydantic import BaseModel ,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Nitesh'
    age: Optional[int] = None
    email:EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5,description='A decimal value representing the cgpa of the student')
    
new_student = {'age':56,'email':'abc@gmail.com'}

student = Student(**new_student)

# Convert into Dict
student_dict = dict(student)

# Convert into Json
student_json = student.model_dump_json()

print(student)