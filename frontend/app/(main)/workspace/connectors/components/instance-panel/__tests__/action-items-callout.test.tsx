import React from 'react';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { screen, fireEvent, cleanup } from '@testing-library/react';
import '@/lib/__tests__/test-i18n';

vi.mock('@/app/components/ui/MaterialIcon', () => ({ MaterialIcon: () => null }));

import { ActionItemsCallout } from '../action-items-callout';
import { installDomShims, renderInTheme, makeInstance } from '../../../__tests__/fixtures';
import type { ConnectorActionItem } from '../../../types';

const SWITCH: ConnectorActionItem = {
  code: 'certificate_sign_in',
  severity: 'warning',
  title: 'Switch to certificate sign-in',
  message: 'Upload a client certificate and its private key.',
  action: 'authenticate',
};

beforeEach(() => installDomShims());
afterEach(() => cleanup());

describe('ActionItemsCallout', () => {
  it('shows each item as the connector words it', () => {
    renderInTheme(<ActionItemsCallout instance={makeInstance({ actionItems: [SWITCH] })} />);

    expect(screen.getByText('Switch to certificate sign-in')).toBeTruthy();
    expect(screen.getByText('Upload a client certificate and its private key.')).toBeTruthy();
  });

  it('opens the Authenticate tab for an item done there', () => {
    const open = vi.fn();
    renderInTheme(<ActionItemsCallout instance={makeInstance({ actionItems: [SWITCH] })} onOpenAuthentication={open} />);

    fireEvent.click(screen.getByRole('button', { name: 'Update authentication' }));

    expect(open).toHaveBeenCalledOnce();
  });

  it('offers no button when nothing is done on the Authenticate tab', () => {
    const elsewhere = { ...SWITCH, code: 'sharepoint_permission_missing', action: null };
    renderInTheme(<ActionItemsCallout instance={makeInstance({ actionItems: [elsewhere] })} onOpenAuthentication={vi.fn()} />);

    expect(screen.queryByRole('button', { name: 'Update authentication' })).toBeNull();
  });

  it('renders nothing when there is nothing to do', () => {
    const { container } = renderInTheme(<ActionItemsCallout instance={makeInstance({ actionItems: [] })} />);

    expect(container.textContent).toBe('');
  });
});
