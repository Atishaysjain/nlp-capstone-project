// src/tests/Home.test.js
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import Home from '../pages/Home';
import axios from 'axios';
import ResumeUpload from '../components/ResumeUpload';
import { BrowserRouter as Router } from 'react-router-dom';

// Mock axios to simulate network request
jest.mock('axios');

describe('Home Component', () => {
    it('should render the Home page and display the upload button', () => {
        render(
            <Router>
                <Home />
            </Router>
        );

        const button = screen.getByRole('button', { name: /Upload Your Resume/i });
        expect(button).toBeInTheDocument();
    });

    it('should handle upload errors gracefully', async () => {
        // Simulate an error in the upload process
        const mockUpload = jest.fn().mockRejectedValue(new Error('Upload failed'));

        // Render the component
        render(<ResumeUpload onUpload={mockUpload} loading={false} error={true} />);

        // Trigger the file upload (e.g., click the upload button)
        // const button = screen.getByRole('button', { name: /Upload Your Resume/i });
        // fireEvent.click(button);

        // Wait for the error message to appear
        const errorMessage = await screen.findByText(/Oops! Something went wrong while processing your resume./i);
        expect(errorMessage).toBeInTheDocument();

        // Check if the retry button is rendered as well
        const retryButton = screen.getByRole('button', { name: /Retry/i });
        expect(retryButton).toBeInTheDocument();
    });
});
