from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import date


class Education(BaseModel):
    degree: str = Field(description="Degree obtained, e.g., Bachelor of Science")
    institution: str = Field(description="Name of the institution")
    start_date: Optional[date] = Field(description="Start date of the education")
    end_date: Optional[date] = Field(description="End date of the education, if applicable")


class WorkExperience(BaseModel):
    job_title: str = Field(description="Title of the job")
    company: str = Field(description="Name of the company")
    start_date: Optional[date] = Field(description="Start date of the job")
    end_date: Optional[date] = Field(description="End date of the job, if applicable")
    description: Optional[str] = Field(description="Description of job responsibilities")


class Skill(BaseModel):
    name: str = Field(description="Name of the skill")
    proficiency: str = Field(default="Basic", description="Proficiency level, e.g., Beginner, Intermediate, Advanced")


class Resume(BaseModel):
    full_name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    summary: str = Field(
        default="Generated summary based on the resume content.",
        description="Summary or objective from the resume",
    )
    education: List[Education] = Field(description="List of educational qualifications")
    work_experience: List[WorkExperience] = Field(description="List of work experiences")
    skills: List[Skill] = Field(description="List of skills")
    links: List[HttpUrl] = Field(description="List of relevant links such as LinkedIn or GitHub profiles")