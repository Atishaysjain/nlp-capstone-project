// src/pages/MatchList.jsx
import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import JobCard from '../components/JobCard';
import JobModal from '../components/JobModal';

export default function MatchList() {
    const { state } = useLocation();
    const navigate = useNavigate();

    const jobs = state?.jobs || [];

    const [selectedJob, setSelectedJob] = useState(null);

    useEffect(() => {
        if (!jobs.length) {
            // Redirect to home if navigated directly without jobs
            navigate('/');
        }
    }, [jobs, navigate]);

    const handleJobClick = (job) => {
        setSelectedJob(job);
    };

    const handleCloseModal = () => {
        setSelectedJob(null);
    };

    return (
        <section className="pt-12 pb-16 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold text-center text-indigo-700 mb-10">Top Job Matches</h1>
                
                <div className="max-h-[75vh] overflow-y-auto px-4 py-4 relative">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 relative z-0 overflow-visible">
                        {jobs.map((job, idx) => (
                            <JobCard
                                key={idx}
                                job={job}
                                onClick={() => handleJobClick(job)}
                            />
                        ))}
                    </div>
                </div>

                {selectedJob && (
                    <JobModal job={selectedJob} onClose={handleCloseModal} />
                )}
            </div>
        </section>
    );
}
