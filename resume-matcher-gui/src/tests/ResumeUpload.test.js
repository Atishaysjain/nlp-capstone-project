// src/tests/ResumeUpload.test.js
import { render, fireEvent, screen } from '@testing-library/react';
import React from 'react';
import ResumeUpload from '../components/ResumeUpload';

describe('ResumeUpload Component', () => {
    it('should render the upload button', () => {
        render(<ResumeUpload onUpload={() => { }} loading={false} />);

        const button = screen.getByRole('button', { name: /Upload Your Resume/i });
        expect(button).toBeInTheDocument();
    });

    it('should show loading spinner when uploading', () => {
        render(<ResumeUpload onUpload={() => { }} loading={true} />);

        const spinner = screen.getByRole('status');
        expect(spinner).toBeInTheDocument();

        const message = screen.getByText(/Analyzing your resume…/i);
        expect(message).toBeInTheDocument();
    });

    it('should handle file selection', () => {
        const onUploadMock = jest.fn();
        render(<ResumeUpload onUpload={onUploadMock} loading={false} />);

        const inputFile = screen.getByTestId('resume-input');
        fireEvent.change(inputFile, {
            target: { files: [new File(['resume'], 'resume.pdf', { type: 'application/pdf' })] },
        });
        expect(onUploadMock).toHaveBeenCalled();

    });
});
