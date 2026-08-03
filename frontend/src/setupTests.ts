frontend/src/setupTests.ts
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders app header', () => {
  render(<App />);
  const headerElement = screen.getByText(/Surface Auto-Annotator/i);
  expect(headerElement).toBeInTheDocument();
});
