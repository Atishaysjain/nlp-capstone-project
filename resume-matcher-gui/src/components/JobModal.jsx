// src/components/JobModal.jsx
import { CSSTransition } from 'react-transition-group';
import { useState, useEffect, useRef } from 'react';

export default function JobModal({ job, onClose }) {
  const [showModal, setShowModal] = useState(false);
  const nodeRef = useRef(null);

  useEffect(() => {
    setShowModal(true);
  }, []);

  const handleOverlayClick = () => {
    setShowModal(false);
    setTimeout(onClose, 250); // match exit duration
  };

  const handleModalClick = (e) => {
    e.stopPropagation();
  };

  return (
    <div
      className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50"
      onClick={handleOverlayClick}
    >
      <CSSTransition
        in={showModal}
        timeout={300}
        classNames="modal"
        unmountOnExit
        nodeRef={nodeRef} // <--- PASS ref
      >
        <div
          ref={nodeRef} // <--- ATTACH ref
          className="bg-white p-8 rounded-lg shadow-lg w-[90%] md:w-[60%] lg:w-[40%] max-h-[75%] overflow-y-auto relative"
          onClick={handleModalClick}
        >
          <button
            onClick={handleOverlayClick}
            className="absolute top-4 right-4 text-gray-600 hover:text-gray-900"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              className="w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>

          <h2 className="text-2xl font-semibold text-blue-800 mb-4">{job.title}</h2>
          <p className="text-gray-600 font-medium">{job.company}</p>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-sm font-semibold text-gray-500">Match Score</span>
            <span className="text-lg font-bold text-green-600">{job.similarity_score}%</span>
          </div>
          <p className="text-gray-500 mt-2">{job.description}</p>
        </div>
      </CSSTransition>
    </div>
  );
}
