// src/pages/Home.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import React from 'react';
import ResumeUpload from '../components/ResumeUpload';

export default function Home() {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // Error state

    const handleUpload = async (file) => {
        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true);
            setError(null);  // Reset error state

            //   const response = await axios.post("http://127.0.0.1:8000/get_matches/", formData, {
            //     headers: { "Content-Type": "multipart/form-data" },
            //   });

            let response = await fetch("/mock_data.json");
            response = await response.json();

            // Simulate delay if needed
            await new Promise(resolve => setTimeout(resolve, 3000));

            console.log(response)

            // throw new Error();

            // Navigate to MatchList page with job data
            navigate("/matches", { state: { jobs: response.data.jobs } });
        } catch (error) {
            console.error("Upload failed:", error);
            setError("Something went wrong while uploading resume."); // Set error message
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-r from-blue-50 to-indigo-100 flex flex-col items-center pt-48">
            <header className="text-center mb-2">
                <h1 className="text-6xl font-extrabold text-blue-900 drop-shadow-md">ResuMatch AI</h1>
                <p className="text-2xl font-semibold text-gray-700 mt-4 mb-6 max-w-lg mx-auto">
                    Smart job matching powered by AI. Find your perfect job match today!
                </p>
            </header>
            <div className="w-full max-w-2xl bg-white rounded-xl shadow-2xl p-8 space-y-8">
                <ResumeUpload onUpload={handleUpload} loading={loading} error={error} />
            </div>
        </div>
    );
}
