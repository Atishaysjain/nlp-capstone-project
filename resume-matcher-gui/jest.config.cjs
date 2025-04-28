module.exports = {
    transform: {
      '^.+\\.(js|jsx)$': 'babel-jest',
    },
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['<rootDir>/jest.setup.js'], // Optional if you want to configure global test setup
    moduleNameMapper: {
      '\\.(css|less|scss|sass)$': 'identity-obj-proxy', // Optional: for handling styles
    },
  };