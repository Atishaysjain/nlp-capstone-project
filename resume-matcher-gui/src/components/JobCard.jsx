// src/components/JobCard.jsx
export default function JobCard({ job, onClick }) {
  return (
    <div
      className="bg-white rounded-2xl shadow-lg p-6 transition-transform transform-gpu hover:scale-105 hover:shadow-2xl hover:z-10 border hover:border-indigo-500 relative mx-2 cursor-pointer"
      onClick={onClick}
    >
      <h2 className="text-2xl font-semibold text-blue-800">{job.title}</h2>
      <p className="text-gray-600 font-medium">{job.company}</p>
      <p className="text-gray-500 mt-2 line-clamp-5 overflow-hidden">{job.description}</p>
      <div className="mt-4 flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-500">Match Score</span>
        <span className="text-lg font-bold text-green-600">{job.similarity_score}%</span>
      </div>
    </div>
  );
}